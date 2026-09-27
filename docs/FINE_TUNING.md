# Fine-tuning Aural One

The public code includes the selected **full acoustic-weight** recipe. It trains real Gemma audio weights while keeping the language model and the released LoRA adapter frozen. The inference model remains a single native-audio Gemma. The [schedule builder](../scripts/build_schedule.py) was run against the original local manifests with seed `20260928` and reproduced the selected schedule's SHA-256 **exactly**: `cc8c4460fa41c78a641871cc9444a7ce7b70db63313a2dc7cbb7535e5c7eed73`. The [aggregate schedule audit](stage_b_schedule_audit.json) records exposures and speaker coverage without distributing source media or row IDs.

The **training script is provided for use with your own licensed local audio**. The original audio, row-level manifests, and optimizer state are not redistributed. The public release adapter is the starting LoRA. Omitting `--start-acoustic` starts acoustic weights from the pinned Gemma base; supplying the published acoustic delta continues from the preview model.

## Data layout

Place these JSONL files under one `DATA_ROOT`:

```text
DATA_ROOT/
  crema_train.jsonl
  subesco_train.jsonl
  typed_train.jsonl
  subesco_train_same_sentence_contrasts.jsonl
  audio/...
```

Each training row needs a unique `id`, local `audio_path`, `state` object, `question`, and 2–8 `options`. Provide either `target_distribution` (nonnegative, sums to one) or `gold_index` (zero-based). The schedule builder also uses:

| Manifest | Additional fields |
|---|---|
| `crema_train.jsonl` | `source_label` among angry/disgusted/fearful/happy/neutral/sad, `actor`, listener-vote `target_distribution` |
| `subesco_train.jsonl` | `speaker_key`, `sentence_id`, `trial`, `perceived_consensus_index` for paired rows |
| `typed_train.jsonl` | `type` among `choice`/`noul`/`score`, and a grouping key `group` |

One generic row:

```json
{"id":"my-audio-001","audio_path":"audio/example.wav","state":{"task":"listen to the voice"},"question":"Which emotion is audible?","options":["angry","happy","neutral"],"target_distribution":[0.0,0.8,0.2],"source_label":"happy","actor":"speaker-001"}
```

The pair manifest uses the two `subesco_train.jsonl` IDs and their correct indices:

```json
{"id":"same-voice-pair-001","left_id":"clip-a","right_id":"clip-b","left_label_index":1,"right_label_index":3,"speaker_key":"speaker-002"}
```

Pairs should use the same speaker and words with **different listener-confirmed vocal emotion**. The trainer checks the IDs and consensus indices, but the dataset curator remains responsible for the acoustic and label validity. Keep development and reserved-test speakers outside all training manifests. The schedule builder checks for train/development ID overlap when `*_dev.jsonl` manifests are present.

## Run the recipe

Use a CUDA GPU and a PyTorch build compatible with it. The selected run used PyTorch 2.13.0+cu130, Transformers 5.17.0, and PEFT 0.21.0. The selected training/evaluation process peaked near **11 GiB PyTorch allocation** on a 32 GB RTX PRO 4500. Allow room for the base model, optimizer, activations, media and checkpoint files.

```bash
hf download google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 --local-dir ./base
hf download blazeofchi/Aural-One-E2B --local-dir ./aural-release

python scripts/build_schedule.py \
  --data-root "$DATA_ROOT" \
  --output "$DATA_ROOT/schedule.jsonl" \
  --audit "$DATA_ROOT/schedule.audit.json" \
  --seed 20260928 --updates 1000

python scripts/train_full_audio.py \
  --model-dir ./base \
  --adapter-dir ./aural-release/adapter \
  --config ./configs/stage_b_paired.yaml \
  --data-root "$DATA_ROOT" \
  --pairs "$DATA_ROOT/subesco_train_same_sentence_contrasts.jsonl" \
  --schedule "$DATA_ROOT/schedule.jsonl" \
  --output ./runs/my-acoustic-run
```

To continue from this preview, add `--start-acoustic ./aural-release/acoustic/acoustic_weights.safetensors`. For a two-update pipeline check, add `--stop-after 2`; resume with `--resume ./runs/my-acoustic-run/checkpoints/step-0002` using the **same** code, config, manifests and schedule. Full checkpoints include BF16 acoustic weights, FP32 optimizer masters, and RNG state. The trainer records SHA-256 identities and rejects mismatched resumes.

The selected schedule used 1,000 eight-example updates: 3,200 CREMA-D, 2,400 SUBESCO, and 2,400 typed exposures, with **500** same-voice paired updates. The code uses listener-vote cross-entropy plus a 0.25-weight softplus crossed-margin loss (margin 0.5), FP32-master AdamW over BF16 forward weights, 20-update warmup, 2e-6 Conformer LR, 5e-6 projection LR, 0.01 weight decay, and gradient clipping at 1.0. It saves checkpoints every 100 updates by default.

## Evaluate before promoting a checkpoint

The generic evaluator reports rows, accuracy, macro-F1 when all options share the same class order, per-class counts, and soft-vote cross-entropy/Brier when targets exist. Use separate held-out manifests for emotion, sounds and typed tasks. For example:

```bash
python scripts/evaluate.py \
  --model-dir ./base \
  --adapter-dir ./aural-release/adapter \
  --config ./configs/stage_b_paired.yaml \
  --acoustic ./runs/my-acoustic-run/checkpoints/step-0100/acoustic_weights.safetensors \
  --manifest "$DATA_ROOT/crema_dev.jsonl" \
  --data-root "$DATA_ROOT" \
  --output ./runs/my-acoustic-run/crema-dev-step100.json
```

Recheck typed decisions and other languages before replacing the published preview. A lower training loss alone does not establish better audio grounding. The original selected model's metrics and known scopes are in [EVALUATION.md](EVALUATION.md).

The public pipeline passed a **synthetic-only GPU smoke** on a 24 GB Blackwell partition: two updates had finite loss and nonzero FP32 gradient norms, step 2 saved a checkpoint, a fresh process resumed at step 3, the weight hashes changed, and the generic evaluator read the step-3 checkpoint. This verifies the code path, **not** model quality or reproduction of the 1,000-update result. The two-update toy inputs can be generated with `python scripts/make_toy_training_data.py /tmp/aural-toy`, followed by the schedule command above with `--updates 3`.
