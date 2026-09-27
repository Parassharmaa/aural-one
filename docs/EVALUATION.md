# Evaluation: Aural One E2B Preview

This page reports the **selected Stage-B step-1,000 checkpoint**, SHA-256 `ef80763236b2467a886d52fba51769de4dcfbdce909dd320803b6d2d2d41db96`. Scores use different datasets, splits, and metrics; none is an overall System One score. One training seed was used. No separate classifier is used at inference.

![Aural One audio evaluation chart: macro-F1 is shown separately from question accuracy](../assets/audio-evaluation.png)

## Emotion and typed decisions

| Fixed evaluation | Frozen step-420 reference | Aural One preview | Scope |
|---|---:|---:|---|
| CREMA-D macro-F1 | 0.361 | **0.391** | 445 actor-held-out development clips |
| CREMA-D listener-vote cross-entropy ↓ | 1.535 | **1.473** | Same development set |
| CREMA-D soft-vote Brier ↓ | 0.263 | **0.232** | Same development set |
| CREMA-D 10-bin soft-vote ECE ↓ | 0.070 | **0.044** | Same development set |
| SUBESCO macro-F1 | 0.134 | **0.295** | 700 clips from two training-unseen development speakers |
| SUBESCO listener-vote cross-entropy ↓ | 2.063 | **1.605** | Same development set |
| SUBESCO soft-vote Brier ↓ | 0.647 | **0.523** | Same development set |
| SUBESCO 10-bin soft-vote ECE ↓ | 0.064 | **0.035** | Same development set |
| Typed choice correct | 188/200 | **189/200** | Fixed development examples |
| Typed No/Null correct | 175/200 | **174/200** | Fixed development examples |
| Typed ordinal score correct | 189/200 | **191/200** | Fixed development examples |

The **SUBESCO reserved test** was opened once after checkpoint selection. On four further unseen speakers, Aural One scored **0.344 macro-F1 over 1,012 unanimously perceived clips** among 1,400 recordings; listener-vote cross-entropy was **1.700** over all 1,400. Per-class F1 on unanimous clips was angry 0.469, disgusted 0.222, fearful 0.047, happy 0.358, neutral 0.581, sad 0.334, surprised 0.394. This is a speaker-held-out acted-emotion result, not a natural-conversation estimate.

On 40 same-speaker, same-words development contrasts, the selected model put the target relative emotion margin in the correct direction on **35/40**, versus **23/40** for the frozen reference. Both clips received the correct top seven-way emotion on **2/40** versus **1/40**. This indicates audio sensitivity while leaving room to improve complete class decisions.

## Other audio checks

| Check | Frozen reference | Aural One preview | Interpretation |
|---|---:|---:|---|
| Previously opened German EmoDB macro-F1 | 0.078 | **0.102** | 195 clips; exploratory transfer, not a blind test |
| Italian Emozionalmente development macro-F1 | **0.232** | 0.203 | 262 unanimous clips, 69 development actors across all 1,202 clips |
| Italian listener-vote cross-entropy ↓ | **2.492** | 2.545 | All 1,202 Italian development clips |
| Original 24-clip sound screen correct | 13/24 | **14/24** | Eight each of baby cry, laugh, cough; source-confounded |
| Draft ~60-second speech questions correct | — | **14/18** | Six development audiobook clips, three questions each |
| New natural 45–60-second MMAU-Pro speech questions correct | 16/39 | **18/39** | 39 distinct held-out waveforms; one question each |

On the small sound screen, Aural One answered baby cry **6/8**, laughter **8/8**, cough **0/8**. A subsequent **blind human review of a different, source-diverse 24-preview candidate set** confirmed 17 clear labels, found six mixed clips and one different sound. Those human-reviewed previews have **not been scored as model accuracy**. A second 14-clip replacement review is prepared. None of this candidate media is bundled with the release.

The 39-clip natural long-speech slice scored **18/39 with the real recording** and **11/39 after audio was swapped**. The two-answer gain over the frozen reference had a paired interval spanning zero; the median maximum choice probability on incorrect real-audio answers was 0.911. The earlier six-clip draft speech slice kept **18/18 top-choice parity** when three question suffixes shared audio processing, though three probability vectors exceeded a predeclared 0.03 parity tolerance (worst 0.0378). These sets establish task-specific behavior, not general 60-second audio understanding.

## Warm serving and cost

The fixed request here was a **57.89-second Opus recording (238 KB JSON)** with **three named choice questions**. Measurements used ten warm calls per single-request route after two warmups, and 12 calls per concurrency condition. The fast path processed shared audio once and right-batched question suffixes.

| Route | Client or pod-local wall p50 / p95 | Model p50 / p95 | Scope |
|---|---:|---:|---|
| RTX PRO 4500, Romania pod-local HTTP | **0.508 / 0.579 s** | 0.242 / 0.290 s | Same pod to authenticated app |
| Tokyo → Romania direct pod proxy | 2.086 / 2.498 s | ~0.24 / ~0.29 s | Internet client end to end |
| Tokyo → Japan H100 direct pod proxy | 1.635 / 1.952 s | ~0.21 s median | Different GPU/runtime and route |

At the Romania proxy, four concurrent clients achieved **2.029 successful requests/s** over 12 fixed requests with **2.211 s client p95** and 0.115 s queue p95; all top-choice vectors matched. At the Japan H100 proxy, four clients achieved **2.368 successful requests/s**, client p95 **1.888 s**, and queue p95 0.048 s over 12 calls. These short bursts are not sustained-capacity benchmarks.

At the observed Romania four-client throughput, allocating the displayed **$0.72/hour GPU price** gives **about $0.000099 per successful request**, or **$0.000061 per 1,000 shared input positions** for this request. This is a GPU-only allocation at sustained throughput; idle, disk, transfer, cold starts and network are excluded. It is not a token-billed price. The Japan H100 allocation was about $0.000409/request at $3.49/hour and the observed 2.368 RPS. The selected Stage-B run reported **11,006 MiB peak allocated GPU memory** during training and evaluation on the 32 GB RTX PRO 4500; minimum inference VRAM was not measured.

The Tokyo end-to-end sub-second target is **still a research target**. These HTTP tests used temporary authenticated pods, not a public serverless endpoint. Cold start, sustained load, broader schema choices, and production p95 remain to be measured.

The **public reference loader** was separately smoke-tested on a 24 GB RTX PRO 6000 Blackwell partition with PyTorch 2.13.0+cu130. It loaded the pinned release files, verified their hashes, and returned valid distributions for three named questions on both a short speech clip and a synthetic 58-second two-chunk audio input. The three sequential forwards took **5.128 s** on the first short run and **3.669 s** on the later 58-second run after model/base caching; PyTorch reported **9,774 MiB allocated**. These are functional smokes, not an accuracy or warm-service benchmark, and this simple loader does not implement the shared-audio fast path measured above.

## Training follow-ups

Two controlled 100-update follow-ups were explored after selecting this release checkpoint. Training all twelve Conformer blocks improved Italian development listener-vote cross-entropy to **2.404** but reached only **0.207** unanimous macro-F1, with sad **0/44** and fearful **1/17** correct; it was not promoted. A separate per-clip top-margin arm reached Italian CE **2.417** and F1 **0.211** but reduced the fixed CREMA-D screen macro-F1 **0.343 → 0.288**; it was rejected. The published weights are the earlier Stage-B checkpoint, not either follow-up.

## Reproducibility notes

The base revision, delta hashes, seed, and update count are pinned in [`release.json`](../release.json). The metric scopes above come from the frozen development, reserved-test, long-speech, and warm-serving runs. Audio and row-level private results are not redistributed here because the source datasets have their own terms. The public code exposes the choice-scoring path; the optimized HTTP endpoint used for timing remains experimental and is not represented as a production deployment.
