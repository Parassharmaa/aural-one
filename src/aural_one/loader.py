"""Load the Aural One preview weights and score named audio questions."""

import hashlib
import json
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor

from .model import choice_token_ids, load_adapter, load_model, logits_for_row

DEFAULT_REPO = "blazeofchi/Aural-One-E2B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AuralOne:
    """Single native-audio Gemma model with conditional choice distributions."""

    def __init__(self, model, processor, answer_ids):
        self.model = model
        self.processor = processor
        self.answer_ids = answer_ids

    @torch.inference_mode()
    def score(self, audio: str | Path, state: dict, questions: dict) -> dict:
        """Return probabilities for 1..32 named questions over 2..8 supplied options.

        Binary No/Yes and ordinal decisions are choices with corresponding
        option labels. These probabilities are conditional on the supplied
        options and are not calibrated confidence estimates.
        """
        audio = Path(audio).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        duration = sf.info(audio).duration
        if not 0 < duration <= 60:
            raise ValueError("Aural One preview accepts recordings over 0 and up to 60 seconds")
        if not isinstance(state, dict) or not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
            raise ValueError("Provide a state object and 1..32 named questions")
        answers = {}
        with tempfile.TemporaryDirectory() as directory:
            if duration <= 30:
                paths = [str(audio)]
            else:
                waveform, rate = sf.read(audio, dtype="float32", always_2d=True)
                mono = waveform.mean(axis=1)
                if rate != 16000:
                    mono = librosa.resample(mono, orig_sr=rate, target_sr=16000)
                mono = mono[:60 * 16000]
                paths = []
                for offset in range(0, len(mono), 30 * 16000):
                    path = Path(directory) / f"part-{len(paths)}.wav"
                    sf.write(path, mono[offset:offset + 30 * 16000], 16000, subtype="PCM_16")
                    paths.append(str(path))
                if len(paths) != 2:
                    raise ValueError("Expected two audio chunks for a recording over 30 seconds")
            for name, spec in questions.items():
                if not isinstance(name, str) or not name or not isinstance(spec, dict):
                    raise ValueError("Each named question must be an object")
                prompt, options = spec.get("question"), spec.get("options")
                if (not isinstance(prompt, str) or not prompt.strip() or
                        not isinstance(options, list) or not 2 <= len(options) <= 8 or
                        any(not isinstance(option, str) or not option.strip() for option in options) or
                        len(set(options)) != len(options)):
                    raise ValueError(f"Question {name!r} needs text and 2..8 distinct nonempty options")
                row = {"audio_path": paths[0], "audio_paths": paths,
                       "state": state, "question": prompt,
                       "options": options, "prompt_layout": "audio_first"}
                logits = logits_for_row(self.model, self.processor, row, "/", self.answer_ids)
                probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                answers[name] = {
                    "choice": options[int(logits.argmax())],
                    "probabilities": {option: round(float(probability), 6)
                                      for option, probability in zip(options, probabilities, strict=True)},
                }
        return {"model": "Aural One E2B Preview", "answers": answers}


def load_aural_one(repo_id: str = DEFAULT_REPO, *, revision: str | None = None,
                   device_map: str = "auto") -> AuralOne:
    """Load the public delta plus the exactly pinned Gemma 4 E2B base."""
    local_release = Path(repo_id).expanduser()
    if local_release.is_dir():
        if revision is not None:
            raise ValueError("revision applies to a Hugging Face repo, not a local folder")
        release_dir = local_release.resolve()
    else:
        release_dir = Path(snapshot_download(repo_id, revision=revision))
    release = json.loads((release_dir / "release.json").read_text())
    if release.get("format") != "aural-one-e2b-preview-v1":
        raise ValueError("Unrecognized Aural One release format")
    base_dir = Path(snapshot_download(release["base_model"], revision=release["base_revision"]))
    if _sha256(base_dir / "model.safetensors") != release["base_weight_sha256"]:
        raise ValueError("Pinned Gemma base weights differ from release.json")
    config_path = release_dir / "configs/stage_b_paired.yaml"
    adapter_path = release_dir / "adapter/adapter.safetensors"
    acoustic_path = release_dir / "acoustic/acoustic_weights.safetensors"
    for key, path in (("config_sha256", config_path), ("adapter_sha256", adapter_path),
                      ("acoustic_sha256", acoustic_path)):
        if _sha256(path) != release[key]:
            raise ValueError(f"Release file hash mismatch: {path.name}")
    config = yaml.safe_load(config_path.read_text())
    if (config["model"]["repo"] != release["base_model"] or
            config["model"]["revision"] != release["base_revision"]):
        raise ValueError("Base model/config identity mismatch")
    train_config = dict(config["train"])
    train_config["device_map"] = device_map
    processor = AutoProcessor.from_pretrained(base_dir)
    model, _ = load_model(base_dir, train_config, train=False)
    load_adapter(model, release_dir / "adapter")
    weights = load_file(str(acoustic_path))
    parameters = dict(model.named_parameters())
    if (sum(value.numel() for value in weights.values()) != release["acoustic_parameter_count"] or
            any(name not in parameters or parameters[name].shape != value.shape or "lora_" in name
                for name, value in weights.items())):
        raise ValueError("Acoustic weight names, shapes, or count differ")
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].device))
    model.eval()
    answer_ids = choice_token_ids(processor, config["model"]["answer_labels"])
    return AuralOne(model, processor, answer_ids)
