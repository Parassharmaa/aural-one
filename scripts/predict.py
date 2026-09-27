"""Run one JSON request against the public Aural One preview checkpoint."""

import argparse
import json
from pathlib import Path

from aural_one import load_aural_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path, help="JSON containing audio, state, and questions")
    parser.add_argument("--repo-id", default="blazeofchi/Aural-One-E2B")
    parser.add_argument("--revision", help="Optional pinned Hugging Face revision")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    model = load_aural_one(args.repo_id, revision=args.revision)
    print(json.dumps(model.score(request["audio"], request["state"], request["questions"]), indent=2))


if __name__ == "__main__":
    main()
