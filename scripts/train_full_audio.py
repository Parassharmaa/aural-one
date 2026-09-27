"""Fine-tune Gemma's acoustic weights with frozen language model and LoRA.

Inputs are caller-provided, locally licensed JSONL manifests and audio files.
The optimizer and crossed-pair objective match the Aural One E2B recipe.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import peft
import torch
import torch.nn.functional as F
import transformers
import yaml
from safetensors.torch import load_file
from transformers import AutoProcessor

from aural_one.full_audio import (
    keep_frozen_lora_deterministic,
    select_full_audio_parameters,
)
from aural_one.full_audio_checkpoint import load_checkpoint, save_checkpoint
from aural_one.full_audio_optim import MasterWeightAdamW
from aural_one.model import choice_token_ids, load_adapter, load_model, logits_for_row


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def target_loss(row: dict, logits: torch.Tensor) -> torch.Tensor:
    target = row.get("target_distribution")
    if target is not None:
        values = torch.tensor(target, dtype=torch.float32, device=logits.device)
        if (len(values) != len(row["options"]) or
                not torch.isfinite(values).all() or
                abs(float(values.sum()) - 1) > 1e-5 or
                (values < 0).any()):
            raise ValueError("Invalid listener-vote distribution: " + row["id"])
        return -(values * F.log_softmax(logits.float(), -1)).sum()
    gold = row["gold_index"]
    if not isinstance(gold, int) or not 0 <= gold < len(row["options"]):
        raise ValueError("Invalid gold_index: " + row["id"])
    return F.cross_entropy(logits[None], torch.tensor([gold], device=logits.device))


def validate_inputs(args, config: dict) -> tuple[dict, dict, list[dict], dict]:
    paths = {source: args.data_root / f"{source}_train.jsonl"
             for source in ("crema", "subesco", "typed")}
    pools = {}
    for source, path in paths.items():
        rows = read_jsonl(path)
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Empty or duplicate training IDs: " + source)
        for row in rows:
            audio = args.data_root / row["audio_path"]
            if (not audio.is_file() or not isinstance(row.get("state"), dict) or
                    not isinstance(row.get("question"), str) or
                    not isinstance(row.get("options"), list) or
                    not 2 <= len(row["options"]) <= 8 or
                    ("target_distribution" not in row and "gold_index" not in row)):
                raise ValueError("Invalid audio/question training row: " + row["id"])
            if row.get("audio_sha256") and sha(audio) != row["audio_sha256"]:
                raise ValueError("Audio hash mismatch: " + row["id"])
            target = row.get("target_distribution")
            if target is not None and (len(target) != len(row["options"]) or
                                       any(not math.isfinite(value) or value < 0 for value in target) or
                                       abs(sum(target) - 1) > 1e-5):
                raise ValueError("Invalid target distribution: " + row["id"])
            gold = row.get("gold_index")
            if gold is not None and (not isinstance(gold, int) or
                                     not 0 <= gold < len(row["options"])):
                raise ValueError("Invalid gold_index: " + row["id"])
        pools[source] = {row["id"]: row for row in rows}
    if len(set().union(*(set(pool) for pool in pools.values()))) != sum(map(len, pools.values())):
        raise ValueError("Training IDs overlap across sources")
    for source, train_rows in pools.items():
        dev_path = args.data_root / f"{source}_dev.jsonl"
        if dev_path.is_file():
            dev = read_jsonl(dev_path)
            if set(train_rows) & {row["id"] for row in dev}:
                raise ValueError("Training/development ID overlap: " + source)
            key = "actor" if source == "crema" else "speaker_key" if source == "subesco" else None
            if key and ({row[key] for row in train_rows.values()} &
                        {row[key] for row in dev}):
                raise ValueError("Training/development speaker overlap: " + source)
    pair_rows = read_jsonl(args.pairs) if args.pairs else []
    pairs = {row["id"]: row for row in pair_rows}
    if len(pairs) != len(pair_rows):
        raise ValueError("Duplicate pair IDs")
    for pair in pairs.values():
        left, right = (pools["subesco"].get(pair[key]) for key in ("left_id", "right_id"))
        if (left is None or right is None or left["id"] == right["id"] or
                pair["left_label_index"] == pair["right_label_index"] or
                left.get("perceived_consensus_index") != pair["left_label_index"] or
                right.get("perceived_consensus_index") != pair["right_label_index"]):
            raise ValueError("Invalid same-voice pair: " + pair["id"])
        if any(left.get(key) != right.get(key) for key in
               ("speaker_key", "sentence_id", "trial")):
            raise ValueError("Pair does not share speaker, words, and trial: " + pair["id"])
    schedule = read_jsonl(args.schedule)
    if (not schedule or len(schedule) > config["train"]["max_updates"] or
            [item["step"] for item in schedule] != list(range(1, len(schedule) + 1))):
        raise ValueError("Schedule steps are missing, reordered, or over budget")
    paired_updates = 0
    exposures = Counter()
    for update in schedule:
        examples = update["examples"]
        if len(examples) != config["train"]["microbatches_per_update"]:
            raise ValueError("Wrong example count at update " + str(update["step"]))
        grouped = {}
        for item in examples:
            source, ident = item["source"], item["id"]
            if source not in pools or ident not in pools[source]:
                raise ValueError("Schedule references an unknown training ID: " + ident)
            exposures[source] += 1
            if "pair_id" in item:
                if source != "subesco":
                    raise ValueError("Only SUBESCO rows may form same-voice pairs")
                grouped.setdefault(item["pair_id"], []).append(item)
        if len(grouped) > 1:
            raise ValueError("At most one pair is allowed per update")
        for pair_id, items in grouped.items():
            pair = pairs.get(pair_id)
            if (pair is None or len(items) != 2 or
                    {item.get("pair_side") for item in items} != {0, 1} or
                    {item["pair_side"]: item["id"] for item in items} !=
                    {0: pair["left_id"], 1: pair["right_id"]}):
                raise ValueError("Incomplete or mismatched scheduled pair: " + pair_id)
            paired_updates += 1
    if config["train"]["pair_weight"] and not paired_updates:
        raise ValueError("Pair loss is enabled but no pairs are scheduled")
    identity = {
        "base_repo": config["model"]["repo"],
        "base_revision": config["model"]["revision"],
        "adapter_sha256": sha(args.adapter_dir / "adapter.safetensors"),
        "starting_acoustic_sha256": sha(args.start_acoustic) if args.start_acoustic else None,
        "config_sha256": sha(args.config), "schedule_sha256": sha(args.schedule),
        "manifest_sha256": {source: sha(path) for source, path in paths.items()},
        "pairs_sha256": sha(args.pairs) if args.pairs else None,
        "trainer_sha256": sha(Path(__file__)),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
    }
    return pools, pairs, schedule, {"identity": identity, "exposures": dict(exposures),
                                     "paired_updates": paired_updates}


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Train on a CUDA GPU, not on the Mac")
    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text())
    train = config["train"]
    pools, pairs, schedule, audit = validate_inputs(args, config)
    args.output.mkdir(parents=True, exist_ok=True)
    identity_file = args.output / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != audit:
            raise ValueError("Existing output belongs to a different run")
    else:
        write_json(identity_file, audit)
    torch.manual_seed(train["seed"])
    torch.cuda.manual_seed_all(train["seed"])
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model, _ = load_model(args.model_dir, train, train=False)
    load_adapter(model, args.adapter_dir)
    if args.start_acoustic:
        saved = load_file(str(args.start_acoustic))
        parameters = dict(model.named_parameters())
        if any(name not in parameters or parameters[name].shape != value.shape
               for name, value in saved.items()):
            raise ValueError("Starting acoustic weight names/shapes differ")
        with torch.no_grad():
            for name, value in saved.items():
                parameters[name].copy_(value.to(parameters[name].device))
    selected = select_full_audio_parameters(model, train["full_audio_last_layers"])
    optimizer = MasterWeightAdamW(model, selected, train["block_lr"],
                                  train["projection_lr"], train["weight_decay"])
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    answer_ids = choice_token_ids(processor, config["model"]["answer_labels"])
    metrics_file = args.output / "metrics.json"
    if args.resume:
        checkpoint = load_checkpoint(args.resume, optimizer, audit["identity"])
        first_step = checkpoint["step"] + 1
        metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else []
        if metrics and metrics[-1]["step"] > checkpoint["step"]:
            raise ValueError("Metrics and resume checkpoint differ")
        if not metrics or metrics[-1]["step"] < checkpoint["step"]:
            if checkpoint["record"]["step"] != checkpoint["step"]:
                raise ValueError("Checkpoint recovery record differs")
            metrics.append(checkpoint["record"])
            write_json(metrics_file, metrics)
    else:
        if metrics_file.exists():
            raise FileExistsError("Existing run requires --resume")
        first_step, metrics = 1, []
    torch.cuda.reset_peak_memory_stats()
    print("TRAINING_START=" + json.dumps({"first_step": first_step,
          "selected_parameters": selected["trainable_total"], **audit["exposures"],
          "paired_updates": audit["paired_updates"]}), flush=True)
    for update in schedule[first_step - 1:]:
        step = update["step"]
        model.train()
        keep_frozen_lora_deterministic(model)
        warm = min(1.0, step / train["warmup_updates"])
        for group in optimizer.optimizer.param_groups:
            rate = train["block_lr"] if group["name"] == "conformer" else train["projection_lr"]
            group["lr"] = rate * warm
        optimizer.begin_step()
        losses, pair_penalties, processed_pairs = [], [], set()
        for item in update["examples"]:
            if "pair_id" in item:
                pair_id = item["pair_id"]
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)
                pair = pairs[pair_id]
                left, right = (pools["subesco"][pair[key]] for key in ("left_id", "right_id"))
                a, b = pair["left_label_index"], pair["right_label_index"]
                left_logits = logits_for_row(model, processor, left, args.data_root, answer_ids)
                right_logits = logits_for_row(model, processor, right, args.data_root, answer_ids)
                left_loss, right_loss = target_loss(left, left_logits), target_loss(right, right_logits)
                crossed = (left_logits[a] - left_logits[b]) - (right_logits[a] - right_logits[b])
                penalty = F.softplus(train["pair_margin"] - crossed)
                loss = (left_loss + right_loss) / len(update["examples"]) + train["pair_weight"] * penalty
                losses.extend((float(left_loss.detach()), float(right_loss.detach())))
                pair_penalties.append(float(penalty.detach()))
            else:
                row = pools[item["source"]][item["id"]]
                loss = target_loss(row, logits_for_row(model, processor, row, args.data_root, answer_ids))
                losses.append(float(loss.detach()))
                loss = loss / len(update["examples"])
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError("Nonfinite loss at update " + str(step))
            loss.backward()
            optimizer.accumulate()
        gradient_norm = optimizer.step(train["grad_clip_norm"])
        record = {"step": step, "train_loss": sum(losses) / len(losses),
                  "pair_penalty": (sum(pair_penalties) / len(pair_penalties)
                                   if pair_penalties else None),
                  "fp32_grad_norm": gradient_norm,
                  "elapsed_s": time.monotonic() - started,
                  "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2}
        print("TRAIN=" + json.dumps(record), flush=True)
        stop = ((args.stop_after is not None and step >= args.stop_after) or
                (args.max_runtime_seconds is not None and
                 record["elapsed_s"] >= args.max_runtime_seconds))
        if step % args.save_every == 0 or stop or step == schedule[-1]["step"]:
            checkpoint = save_checkpoint(args.output / "checkpoints", step, optimizer,
                                         {"identity": audit["identity"], "record": record})
            metrics.append(record)
            write_json(metrics_file, metrics)
            print("CHECKPOINT=" + str(checkpoint), flush=True)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--start-acoustic", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    options = parser.parse_args()
    if options.save_every <= 0:
        parser.error("--save-every must be positive")
    main(options)
