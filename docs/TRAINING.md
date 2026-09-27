# Training and data

Aural One E2B Preview uses [Gemma 4 E2B Instruct](https://huggingface.co/google/gemma-4-E2B-it) at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The release is a delta: a frozen prior adapter plus 54,294,272 updated weights in the final two audio Conformer blocks, audio output projection, and audio-to-language projection. The language model weights were frozen in this stage.

The selected run used seed `20260928`, **1,000 updates** with eight examples per update and a fixed audio/question schedule. Its 8,000 exposures comprised 3,200 CREMA-D perceived-emotion, 2,400 SUBESCO perceived-emotion, and 2,400 typed-decision examples. Half of the SUBESCO training updates included same-speaker, same-words recordings with different perceived emotion. Supervision combined listener-vote cross-entropy with a crossed audio/option margin (weight 0.25, margin 0.5). The run used BF16 forward weights, FP32 master AdamW, Conformer learning rate 2e-6, projection learning rate 5e-6, weight decay 0.01, and gradient norm clipping at 1.0.

The training data sources were:

| Source | Role | Source terms |
|---|---|---|
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Crowd-voted acted English emotion audio | Dataset: ODbL 1.0; individual contents: DBCL 1.0 |
| [SUBESCO](https://zenodo.org/records/4526477) | Listener-voted acted Bangla emotion audio | CC BY 4.0 |
| Project typed spoken-question/intent examples | Structured-choice retention | Project research examples; no raw examples are included in this release |

The audio files and speaker-level manifests are **not included** in either public repository. Evaluation datasets have their own terms. For source-disjoint speaker protocols, sample counts, and external checks, see [EVALUATION.md](EVALUATION.md).

The published delta is SHA-256 `ef80763236b2467a886d52fba51769de4dcfbdce909dd320803b6d2d2d41db96` for the acoustic weights and `e2b53154b40cd67faf3c9a57226f09c187b060569a7894ee2b3630a4e88937c3` for the adapter. `release.json` pins the base, file hashes, update count, and seed. The base weights are downloaded directly from Google's repository.
