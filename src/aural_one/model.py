"""One-pass native-audio choice scoring on Gemma 4 E2B."""

import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor


def messages_for_row(row, data_root):
    choices = "\n".join(f"{letter}. {option}" for letter, option in
                        zip("ABCDEFGHIJ"[:len(row["options"])], row["options"], strict=True))
    shared_text = ("Read the written state and listen to the audio. Answer the question "
                   "with exactly one letter.\n"
                   f"State: {json.dumps(row['state'], ensure_ascii=False)}\n")
    question_text = f"Question: {row['question']}\nChoices:\n{choices}\nAnswer:"
    prompt = shared_text + question_text
    audio_paths = row.get("audio_paths") or [row["audio_path"]]
    audio_items = [{"type": "audio", "audio": str(Path(data_root) / path)}
                   for path in audio_paths]
    text_item = {"type": "text", "text": prompt}
    layout = row.get("prompt_layout", "audio_first" if row.get("audio_first")
                     else "text_first")
    if layout == "text_first":
        content = [text_item] + audio_items
    elif layout == "audio_first":
        content = audio_items + [text_item]
    elif layout == "state_audio_question":
        content = ([{"type": "text", "text": shared_text}] + audio_items +
                   [{"type": "text", "text": question_text}])
    else:
        raise ValueError(f"Unsupported prompt layout: {layout}")
    return [{"role": "user", "content": content}]


def prepare(processor, row, data_root, label_ids):
    messages = messages_for_row(row, data_root)
    batch = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    assert batch["input_ids"].shape[0] == 1
    assert "input_features" in batch and batch["input_features"].numel() > 0
    return batch


def choice_token_ids(processor, labels):
    ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
    assert all(len(x) == 1 for x in ids), ids
    ids = [x[0] for x in ids]
    assert len(set(ids)) == len(ids), ids
    return ids


def load_model(model_dir, config, train):
    dtype = getattr(torch, config.get("inference_dtype", "bfloat16"))
    attention = config.get("attn_implementation")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=dtype, device_map=config.get("device_map", "auto"),
        low_cpu_mem_usage=True,
        **({"attn_implementation": attention} if attention else {}))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    decoder_pattern = re.compile(
        r"model\.language_model\.layers\.\d+\.self_attn\."
        r"(?:q_proj|v_proj)$|model\.embed_audio\.embedding_projection$")
    decoder_targets = [name for name, _ in model.named_modules()
                       if decoder_pattern.fullmatch(name)]
    if not decoder_targets:
        raise RuntimeError("Gemma decoder/bridge LoRA targets were not found")
    audio_targets = []
    last_audio_layers = int(config.get("audio_lora_last_layers", 0))
    if last_audio_layers:
        audio_pattern = re.compile(
            r"model\.audio_tower\.layers\.(\d+)\.self_attn\."
            r"(?:q_proj|k_proj|v_proj|post)\.linear$")
        candidates = [(name, int(match.group(1)))
                      for name, _ in model.named_modules()
                      if (match := audio_pattern.fullmatch(name))]
        layer_ids = sorted({index for _, index in candidates})
        if len(layer_ids) < last_audio_layers:
            raise RuntimeError(f"Expected {last_audio_layers} audio attention layers; found {layer_ids}")
        selected = set(layer_ids[-last_audio_layers:])
        audio_targets = [name for name, index in candidates if index in selected]
        if len(audio_targets) != 4 * last_audio_layers:
            raise RuntimeError(f"Incomplete audio attention targets: {audio_targets}")
    targets = decoder_targets + audio_targets
    audio_rank = int(config.get("audio_lora_rank", config["lora_rank"]))
    audio_alpha = int(config.get("audio_lora_alpha", config["lora_alpha"]))
    lora = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=0.05,
        target_modules=targets,
        rank_pattern={"^" + name: audio_rank for name in audio_targets},
        alpha_pattern={"^" + name: audio_alpha for name in audio_targets},
        bias="none",
    )
    inject_adapter_in_model(lora, model)
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    if any("lora_" not in name for name in trainable_names):
        raise RuntimeError("Unexpected trainable base parameter")
    if last_audio_layers and not any("audio_tower" in name for name in trainable_names):
        raise RuntimeError("Audio tower adapters are not trainable")
    if train and config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
    print(f"Trainable LoRA parameters: {trainable:,}; audio modules: {len(audio_targets)}", flush=True)
    return model, lora


def logits_for_row(model, processor, row, data_root, label_ids, with_token_count=False,
                   use_cache=False):
    option_count = len(row["options"])
    if not 2 <= option_count <= len(label_ids):
        raise ValueError(f"Expected 2..{len(label_ids)} options, got {option_count}")
    features = prepare(processor, row, data_root, label_ids)
    token_count = int(features["attention_mask"].sum())
    features = {k: v.to(model.device) for k, v in features.items()}
    out = model(**features, logits_to_keep=1, use_cache=use_cache)
    logits = out.logits[0, -1, label_ids[:option_count]].float()
    return (logits, token_count) if with_token_count else logits


def save_adapter(model, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in
               get_peft_model_state_dict(model).items()}
    save_file(tensors, path / "adapter.safetensors")
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_adapter(model, path):
    from peft import set_peft_model_state_dict

    state = load_file(str(Path(path) / "adapter.safetensors"))
    set_peft_model_state_dict(model, state)
