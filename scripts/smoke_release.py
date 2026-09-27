"""Smoke-test a release delta on a GPU using a caller-provided audio file."""

import argparse
import json
import math
import time

import torch

from aural_one import load_aural_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="blazeofchi/Aural-One-E2B")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This is a GPU smoke test")
    started = time.perf_counter()
    model = load_aural_one(args.release)
    loaded_s = time.perf_counter() - started
    questions = {
        "emotion": {"question": "Which emotion is expressed in the speaker's voice?",
                    "options": ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]},
        "audible_speech": {"question": "Is a person speaking?", "options": ["No", "Yes"]},
        "intensity": {"question": "How strong is the vocal emotion?", "options": ["0", "1", "2", "3", "4"]},
    }
    started = time.perf_counter()
    answer = model.score(args.audio, {"task": "listen to the voice"}, questions)
    score_s = time.perf_counter() - started
    for name, spec in questions.items():
        result = answer["answers"][name]
        values = result["probabilities"]
        if (result["choice"] not in spec["options"] or set(values) != set(spec["options"]) or
                not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()) or
                abs(sum(values.values()) - 1) > 1e-4):
            raise AssertionError("Invalid choice distribution: " + name)
    report = {"passed": True, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "load_s": round(loaded_s, 3),
              "score_three_questions_s": round(score_s, 3),
              "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
              "answers": answer["answers"]}
    with open(args.output, "w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
