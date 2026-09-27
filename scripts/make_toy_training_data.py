"""Create synthetic WAV/JSONL fixtures for a pipeline smoke, never for quality claims."""

import argparse
import json
import math
import struct
import wave
from pathlib import Path

EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad")
BANGLA_EMOTIONS = (*EMOTIONS, "surprised")


def audio(path: Path, frequency: float) -> None:
    samples = [int(8000 * math.sin(2 * math.pi * frequency * i / 16000))
               for i in range(16000)]
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    (root / "audio").mkdir(parents=True, exist_ok=True)
    crema = []
    for index, label in enumerate(EMOTIONS):
        name = f"crema-{index}.wav"
        audio(root / "audio" / name, 220 + index * 35)
        crema.append({"id": f"demo-crema-{index}", "audio_path": f"audio/{name}",
                      "state": {"task": "hear the vocal emotion"},
                      "question": "Which vocal emotion is audible?", "options": EMOTIONS,
                      "target_distribution": [float(i == index) for i in range(6)],
                      "source_label": label, "actor": f"demo-actor-{index}"})
    subesco = []
    for index in range(2):
        name = f"subesco-{index}.wav"
        audio(root / "audio" / name, 410 + index * 70)
        subesco.append({"id": f"demo-subesco-{index}", "audio_path": f"audio/{name}",
                        "state": {"task": "hear the vocal emotion"},
                        "question": "Which vocal emotion is audible?", "options": BANGLA_EMOTIONS,
                        "target_distribution": [float(i == index) for i in range(7)],
                        "speaker_key": "demo-speaker", "sentence_id": "demo-sentence",
                        "trial": 1, "perceived_consensus_index": index})
    typed = []
    for index, (typ, options, gold) in enumerate((
            ("choice", ["left", "right", "neither"], 0),
            ("noul", ["No", "Yes"], 1),
            ("score", ["0", "1", "2", "3", "4"], 2))):
        name = f"typed-{index}.wav"
        audio(root / "audio" / name, 620 + index * 55)
        typed.append({"id": f"demo-typed-{index}", "audio_path": f"audio/{name}",
                      "state": {"task": "choose a structured answer"},
                      "question": "Choose the appropriate answer for this synthetic example.",
                      "options": options, "gold_index": gold,
                      "type": typ, "group": "demo"})
    pair = [{"id": "demo-pair", "left_id": subesco[0]["id"],
             "right_id": subesco[1]["id"], "left_label_index": 0,
             "right_label_index": 1, "speaker_key": "demo-speaker"}]
    for name, rows in (("crema_train", crema), ("subesco_train", subesco),
                       ("typed_train", typed),
                       ("subesco_train_same_sentence_contrasts", pair)):
        write_rows(root / f"{name}.jsonl", rows)
    print(json.dumps({"directory": str(root), "crema": len(crema),
                      "subesco": len(subesco), "typed": len(typed),
                      "pairs": len(pair), "synthetic_only": True}))


if __name__ == "__main__":
    main()
