"""Check the distributable Aural One delta without loading the base model."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    root = args.release_dir
    meta = json.loads((root / "release.json").read_text())
    expected = {
        "config_sha256": root / "configs/stage_b_paired.yaml",
        "adapter_sha256": root / "adapter/adapter.safetensors",
        "acoustic_sha256": root / "acoustic/acoustic_weights.safetensors",
    }
    for field, path in expected.items():
        if sha(path) != meta[field]:
            raise ValueError(f"Hash mismatch: {path}")
    with safe_open(expected["acoustic_sha256"], framework="np") as file:
        count = 0
        keys = file.keys()
        for name in keys:
            size = 1
            for dimension in file.get_slice(name).get_shape():
                size *= dimension
            count += size
            if not name.startswith(("model.audio_tower.layers.",
                                    "model.audio_tower.output_proj.",
                                    "model.embed_audio.embedding_projection.")):
                raise ValueError(f"Unexpected acoustic weight: {name}")
    if count != meta["acoustic_parameter_count"]:
        raise ValueError("Acoustic parameter count mismatch")
    print(json.dumps({"verified": True, "acoustic_tensors": len(keys),
                      "acoustic_parameters": count, "base_revision": meta["base_revision"]}, indent=2))


if __name__ == "__main__":
    main()
