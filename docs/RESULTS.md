# xiaojev — Full Results

All numbers below are computed by the scripts in this repository and stored as
raw JSON under `results/`. File names are given for each table so every cell
can be traced back to an artifact.

Notes on names:
- **v1** = xiaojev v1, Qwen3-0.6B + scoring head, trained 1200 steps on the
  90k probability tasks only.
- **v2** = xiaojev v2, same architecture, trained 2000 steps from scratch on a
  1:1 mix of probability tasks and NanoJev soft game decisions.
- **vcdm** is the internal codename for the xiaojev checkpoints; it appears in
  raw artifacts (`results/compare_*.json`, engine names in `comparison/`).
- Zero-shot = untuned Qwen3-0.6B, candidates scored by label-token logprob
  softmax (the offline equivalent of the Choice primitive).

## 1. Zero-shot probes of a general LLM (Qwen3.8-27B-AWQ-INT4)

Served locally with vLLM. Verbalized = model generates a percentage at
temperature 0. Token-channel = per-candidate label logprobs, softmaxed
(`probes/systemone.py`). Raw: `results/verbalized_summary.json`,
`results/novel_summary.json`, `results/probe_summary.json`,
`results/elicitation_summary.json`.

### 1a. Verbalized channel — near-perfect calibration

| Set | n | Metric | Value |
|---|---|---|---|
| 10 binary textbook mechanisms × 2 reps | 20 | mean abs err | **0.0046** |
| 5 K-way distributions × 2 reps | 10 | mean TV | **0.000** |
| 10 binary *novel* mechanisms × 2 reps | 20 | mean abs err | **0.0322** |
| 4 K-way *novel* distributions × 2 reps | 6 | mean TV | **0.000** |

Includes non-textbook numbers the model cannot have memorized, e.g.
"two biased coins, P(heads)=0.63 and 0.41, both heads": reference
0.63 × 0.41 = **0.2583**, model answers **0.2583** (4-digit exact).

### 1b. Token-selection channel — badly distorted (same cases, 3 reps each)

| Case | Reference | Model | Error |
|---|---|---|---|
| fair coin, **Choice** | 0.50 / 0.50 | 0.953 / 0.047 | TV 0.453 |
| fair coin, **Noul** (same state!) | P(heads)=0.50 | P=0.053 | abs err 0.447 |
| biased coin 0.7, Choice | 0.70 / 0.30 | 0.980 / 0.020 | TV 0.280 |
| biased coin 0.7, Noul | P=0.70 | P=0.531 | abs err 0.169 |
| lottery K=2 | uniform | 0.777 / 0.223 | TV 0.277 |
| lottery K=5 | uniform | — | TV 0.364 |
| lottery K=5, **reversed options** | uniform | — | TV 0.269 (position bias) |
| lottery K=20 | uniform | — | TV 0.545 |
| weighted lottery 0.7/0.2/0.1 | — | — | TV 0.258 |
| arithmetic 2+2 (deterministic) | 1.0 / 0.0 | 0.981 / 0.019 | TV 0.019 |

Cross-primitive contradiction on the *same* fair-coin state: Choice puts 0.953
on heads while Noul says P(heads) = 0.053 — a gap of ~0.9. The model *knows*
the probability (verbalized channel, 1a) but cannot express it through token
choice.

### 1c. It is not a prompting problem (fair coin / 70% coin / K=5 lottery)

| Case | answer_words | labels_permavg | verbalized | sampling_freq |
|---|---|---|---|---|
| fair coin (err) | 0.442 | 0.211 | **0.000** | 0.460 |
| 70% coin (err) | 0.298 | 0.283 | **0.000** | 0.300 |
| lottery K=5 (TV) | 0.605 | 0.254 | — | 0.400 |

Five elicitation methods; only the verbalized channel is calibrated.

## 2. The closed-source Jev API is miscalibrated too

Data: NanoJev's public probes
([probability semantics](https://github.com/TianyuCodings/NanoJev/blob/main/research/probability_semantics_zh.md),
40-case distribution probe + 8-request coin/Noul control, cost $0.0044),
independently reproduced by us on the frozen Jev receipts, and consistent
with our own 27B probe above.

| Declared P(heads) | Jev Choice P(heads) | Jev Noul P(heads) | Explicit-prior identification |
|---|---|---|---|
| 0.50 (4 requests) | **0.89–0.95** | 0.48–0.49 | 4/4 chose "50%" |
| 0.70 (4 requests) | 0.99–1.00 | 0.69 | 4/4 chose "70%" |

Same binary event, two primitives, answers ~0.4 apart — while the model
correctly identifies the stated prior when asked as a fact. Also: K=2 uniform
lottery returns max item 0.95/0.96 (reference 0.5); K=255 uniform returns
0.75/0.84 per item (reference 1/255 ≈ 0.004).

## 3. Programmatic ground-truth supervision transfers calibration (v1)

90k programmatically generated probability questions with exact analytic
target distributions (`data/datagen.py`, seed 20260921). Metrics on the held
out splits; consistency = mean |P_choice(first) − P_noul(yes)| over paired
Choice/Noul rows of the same binary event. Raw: `results/eval_v1_*.json`.

| Model | Split | n | Acc | NLL | Brier | TV | ECE10 | Consistency |
|---|---|---|---|---|---|---|---|---|
| zero-shot 0.6B | test | 8145 | 0.497 | 1.642 | 0.600 | **0.471** | 0.358 | 0.754 |
| **v1** | test | 8145 | **0.896** | **0.504** | **0.074** | **0.082** | 0.113 | **0.024** (2170 pairs) |
| zero-shot 0.6B | ood | 8896 | 0.569 | 1.328 | 0.386 | 0.353 | 0.275 | 0.604 |
| **v1** | ood | 8896 | **0.719** | **0.662** | **0.058** | **0.120** | 0.063 | 0.080 |
| v1 | dev | 7829 | 0.898 | 0.498 | 0.073 | 0.083 | 0.117 | 0.024 |

OOD = `card` and `dice_sum`, mechanism families never seen in training
(true generalization, not interpolation):

| OOD cell | n | Acc | TV | Brier |
|---|---|---|---|---|
| card / simple | 5000 | 0.678 | 0.075 | **0.019** |
| dice_sum / enumerative | 3896 | 0.770 | 0.179 | **0.108** |

For reference, the Jev API puts 0.89–0.95 on a fair coin (section 2); v1's
test TV of 0.082 over 8145 questions — including fair coins — is far better
calibrated than the API it was never distilled from.

## 4. Specialists do not transfer (bidirectional cross-check with NanoJev)

Game domain: NanoJev's frozen 548-case cohort (274 test + 274 OOD), their
seeded greedy controller (epsilon 0.1, seed 17), every transition re-verified
with NanoJev's own verifier. Raw: `results/compare_v1_frozen.json`,
`results/compare_reverse_nanojev.json`, `results/compare_sanity.json`.

| Direction | Model | Metric | Score | Shared-base reference |
|---|---|---|---|---|
| ours → game | **v1** | macro success (test / ood) | **11.2% / 0.8%** | native untuned Qwen: 15.4% / 12.2% |
| theirs → probability | **NanoJev** (public weights) | accuracy on our test split | **38.5%** (TV 0.463, Brier 0.510) | zero-shot Qwen3-0.6B: **49.7%** |

Each specialist drops *below the shared base model* on the other domain. The
architecture is not magic — generalization comes from data coverage, not from
the scoring-head design. (Sanity check on the frozen artifacts themselves:
every published cell and macro recomputes exactly; 11948/19934/19818
controller transitions verified for selected/jev/native;
`results/compare_sanity.json` → `"passed": true`.)

## 5. Mixed-domain training = a general System One (v2)

v2: 2000 steps, 1:1 mix of probability tasks and NanoJev soft game decisions,
from scratch. Raw: `results/eval_v2_*.json`, `results/compare_v2_frozen.json`,
`results/latency_bench.json`.

### Probability domain (test split, n=8145)

| Model | Acc | NLL | Brier | TV | ECE10 | Consistency |
|---|---|---|---|---|---|---|
| zero-shot 0.6B | 0.497 | 1.642 | 0.600 | 0.471 | 0.358 | 0.754 |
| **v2** | **0.881** | 0.525 | 0.087 | 0.099 | 0.111 | 0.027 |
| v1 (specialist) | 0.896 | 0.504 | 0.074 | 0.082 | 0.113 | 0.024 |
| NanoJev | 0.385 | 1.139 | 0.510 | 0.463 | 0.281 | 0.374 |

v2 OOD (card/dice_sum, n=8896): acc 0.642, Brier 0.085, TV 0.137, ECE 0.094.

### Game domain (frozen 548-case cohort, macro success)

| Model | test | ood | test/maze | test/snake | test/basic | test/predict_position |
|---|---|---|---|---|---|---|
| native Qwen (frozen) | 15.4% | 12.2% | 2/10 | 0/8 | 56/128 | 11/128 |
| **v2** | **48.8%** | **28.7%** | **6/10** | 5/8 | 51/128 | 10/128 |
| NanoJev (frozen) | 66.8% | 45.5% | 4/10 | 8/8 | 128/128 | 27/128 |
| Jev API (frozen) | 65.4% | 43.7% | 7/10 | 8/8 | 56/128 | 11/128 |
| v1 (specialist) | 11.2% | 0.8% | 3/10 | 0/8 | 7/128 | 2/128 |

Paired McNemar v2 vs Jev, all eight split×category cells: **p > 0.16**
(min p = 0.167, ood/predict_position) — no significant difference from the
closed-source API on the game cohort, while v2 beats NanoJev on test Maze
6/10 vs 4/10. (NanoJev keeps a large lead on the Doom basic/predict_position
cells it was specialized on; v2's probability-domain calibration is far
beyond both, section 3 table above.)

### Latency (single RTX 3090, fp32 storage, bf16 autocast forward, warm)

| Engine | Single decision, probability (p50) | Single decision, game (p50) | 24-question packed batch, probability | 24-question packed batch, game |
|---|---|---|---|---|
| **v2** | **45.9 ms** (p95 50.3) | **84.8 ms** (p95 124.9) | 454.5 ms → 52.8 q/s | 2393.5 ms → 10.0 q/s |
| v1 | 47.6 ms | 83.5 ms | 457.4 ms → 52.5 q/s | 2384.4 ms → 10.1 q/s |
| NanoJev (same GPU) | 56.7 ms | 84.5 ms | 284.6 ms → 84.3 q/s | 2039.3 ms → 11.8 q/s |

NanoJev's published A100 reference (forward-only, different GPU/protocol):
p50 84.72 ms ≈ 283 q/s — included in `results/latency_bench.json` for
context, not as an apples-to-apples number.

## Reproducibility notes

- NanoJev rerun fidelity: re-executing the public NanoJev weights through our
  harness is *not* bit-exact (torch 2.10 lacks `torch._native.triton_utils`
  used in their original runs, and batching differs; closed-loop dynamics
  amplify bf16-level drift). Our rerun reaches macro 57.2% / 39.0% vs their
  frozen 66.8% / 45.5%, decision-0 argmax agreement 0.737, episode outcome
  match 0.810 (`results/compare_nanojev_rerun_eval.json`). All our headline
  comparisons therefore use their *frozen published trajectories*, recomputed
  from artifacts, not our rerun.
- The consolidated summary with per-case McNemar win/loss case ids is in
  `results/compare_final.json`.
- Per-row prediction files (`*.preds.jsonl`), rollout trajectories
  (`*episodes*.jsonl`) and journals are reproducible with the scripts but not
  committed (size).
