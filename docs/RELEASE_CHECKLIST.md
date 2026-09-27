# Public preview release review

Checked against the [Hugging Face model release checklist](https://huggingface.co/docs/hub/model-release-checklist) and [GitHub repository best practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories).

| Check | Result |
|---|---|
| One model variant per Hub repo | **Pass** — `blazeofchi/Aural-One-E2B` contains the v0.1.0 preview delta only. |
| Safe weight format and identity | **Pass** — adapter and acoustic weights are safetensors; Hub-reported SHA-256 matches [`release.json`](../release.json). The acoustic file has 47 tensors and 54,294,272 parameters. |
| Base and license | **Pass** — Gemma 4 E2B revision is pinned; Apache 2.0 LICENSE and modified-weight NOTICE are included. Base weights are downloaded separately. |
| Model card metadata | **Pass** — Hugging Face `ModelCard.validate()` accepted license, base model, task, language, and dataset fields. |
| Intended use, training data, and evaluation | **Pass** — card and linked reports give intended research use, source licenses, training settings, quantitative metrics, sample counts, and concise transfer limitations. |
| Audio visualization | **Pass** — chart uses separate emotion macro-F1 and question-accuracy panels, with counts, [data](../assets/audio_eval_chart_data.json), and [render script](../scripts/render_audio_eval.py) in GitHub. Visual output was inspected. |
| Working inference code | **Pass** — public-Hub anonymous GPU load returned the same distributions as local release files. Short and synthetic 58-second three-question inputs passed format checks on a 24 GB GPU partition; see [validation](RELEASE_VALIDATION.md). |
| Building on the checkpoint | **Pass** — public trainer, FP32-master optimizer, schedule builder, checkpoint/resume, evaluator, toy data generator, and [fine-tuning guide](FINE_TUNING.md) are present. The builder reproduced the selected schedule's original SHA-256; synthetic GPU updates saved and resumed. |
| Repository hygiene | **Pass** — GitHub is public under `Parassharmaa`; README, LICENSE, NOTICE, citation, contribution guide, and passing static/schedule CI are present. A staged-file scan found no credentials, private filesystem paths, raw training/evaluation audio, or unrelated project names. |
| Public Hub state | **Pass** — an anonymous API read confirms the model repo is public, and its published safetensors hashes match the tested files. |

The GPU smokes verify **packaging and code paths**, not a new quality result. Full metric scopes and future evaluation work remain in [EVALUATION.md](EVALUATION.md). No public serverless endpoint is claimed in this release.
