"""Score a local held-out JSONL manifest with a Gemma/Aural checkpoint."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoProcessor

from aural_one.model import choice_token_ids, load_adapter, load_model, logits_for_row


def score_group(rows: list[dict]) -> dict:
    labeled = [row for row in rows if row["gold"] is not None]
    correct = sum(row["top"] == row["gold"] for row in labeled)
    same_options = len({tuple(row["options"]) for row in labeled}) == 1
    class_count = len(labeled[0]["options"]) if labeled and same_options else 0
    f1s = []
    by_class = {}
    for label in range(class_count):
        tp = sum(row["gold"] == label and row["top"] == label for row in labeled)
        fp = sum(row["gold"] != label and row["top"] == label for row in labeled)
        fn = sum(row["gold"] == label and row["top"] != label for row in labeled)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        by_class[str(label)] = {"support": tp + fn, "correct": tp, "f1": f1}
        f1s.append(f1)
    vote_rows = [row for row in rows if row["target_distribution"] is not None]
    ce = brier = None
    if vote_rows:
        ce = sum(-sum(target * math.log(max(prob, 1e-12))
                      for target, prob in zip(row["target_distribution"], row["probabilities"], strict=True))
                 for row in vote_rows) / len(vote_rows)
        brier = sum(sum((prob - target) ** 2 for target, prob in
                        zip(row["target_distribution"], row["probabilities"], strict=True))
                    for row in vote_rows) / len(vote_rows)
    return {"rows": len(rows), "labeled_rows": len(labeled), "correct": correct,
            "accuracy": correct / len(labeled) if labeled else None,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "by_class_index": by_class, "vote_rows": len(vote_rows),
            "soft_vote_ce": ce, "soft_vote_brier": brier}


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluate on a CUDA GPU, not on the Mac")
    config = yaml.safe_load(args.config.read_text())
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model, _ = load_model(args.model_dir, config["train"], train=False)
    load_adapter(model, args.adapter_dir)
    if args.acoustic:
        weights = load_file(str(args.acoustic))
        parameters = dict(model.named_parameters())
        if any(name not in parameters or parameters[name].shape != value.shape
               for name, value in weights.items()):
            raise ValueError("Acoustic weights differ from the model")
        with torch.no_grad():
            for name, value in weights.items():
                parameters[name].copy_(value.to(parameters[name].device))
    model.eval()
    tokens = choice_token_ids(processor, config["model"]["answer_labels"])
    source_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not source_rows or len({row["id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Evaluation manifest is empty or contains duplicate IDs")
    results = []
    with torch.inference_mode():
        for row in source_rows:
            logits = logits_for_row(model, processor, row, args.data_root, tokens)
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            gold = row.get("gold_index")
            if gold is None:
                gold = row.get("perceived_consensus_index")
            if gold is not None and (not isinstance(gold, int) or not 0 <= gold < len(row["options"])):
                raise ValueError("Invalid gold index: " + row["id"])
            target = row.get("target_distribution")
            if target is not None and (len(target) != len(probabilities) or
                                       abs(sum(target) - 1) > 1e-5):
                raise ValueError("Invalid target distribution: " + row["id"])
            results.append({"id": row["id"], "source": row.get("source", "all"),
                            "type": row.get("type"), "actor": row.get("actor", row.get("speaker_key")),
                            "options": row["options"], "gold": gold,
                            "top": int(logits.argmax()), "probabilities": probabilities,
                            "target_distribution": target})
    groups = defaultdict(list)
    for row in results:
        groups[row["source"]].append(row)
        if row["type"]:
            groups[row["source"] + "/" + row["type"]].append(row)
    report = {"manifest": str(args.manifest), "checkpoint": str(args.acoustic) if args.acoustic else None,
              "total": score_group(results),
              "by_group": {name: score_group(rows) for name, rows in sorted(groups.items())},
              "rows": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"total": report["total"],
                      "by_group": {name: {k: value[k] for k in ("rows", "accuracy", "macro_f1")}
                                   for name, value in report["by_group"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acoustic", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
