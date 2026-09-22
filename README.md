# xiaojev

**Open, calibrated System One-style decision models.** Give xiaojev a state
plus a dynamic candidate set, and a single forward pass returns a full
probability distribution over the candidates — it *chooses*, it does not
generate text. Think of it as a small, open, locally-runnable analogue of the
[Choice primitive](https://docs.typesafe.ai/primitives/choice), trained to be
calibrated by programmatic ground truth instead of teacher imitation.

[中文文档](README.zh-CN.md) · [Full results](docs/RESULTS.md)

## Headline numbers

| | **xiaojev v2** (0.6B) | Jev (closed API) | NanoJev (open) | Zero-shot Qwen3-0.6B |
|---|---|---|---|---|
| Probability test TV ↓ | **0.099** | miscalibrated (fair coin → 0.89–0.95) | 0.463 | 0.471 |
| Probability test acc ↑ | **0.881** | — | 0.385 | 0.497 |
| Cross-primitive consistency ↓ | **0.027** | ~0.4 Choice-vs-Noul gap | 0.374 | 0.754 |
| Game macro success (test / ood) | **48.8% / 28.7%** | 65.4% / 43.7% | 66.8% / 45.5% | 15.4% / 12.2% |
| vs Jev on games (paired McNemar) | all 8 cells **p > 0.16** | — | — | — |
| Latency, single decision | **46–85 ms** (1× RTX 3090) | network API call | 57–84 ms (same GPU) | — |
| Cost per decision | **$0 (local)** | paid API | $0 (local) | $0 (local) |

v2 is one model that is simultaneously far better calibrated than Jev in the
probability domain and statistically indistinguishable from it on the game
cohort. Details and every intermediate number: [docs/RESULTS.md](docs/RESULTS.md).

## Why this exists — five findings

**1. The knowing–saying gap.** A general LLM (Qwen3.8-27B) is nearly perfectly
calibrated when it *verbalizes* probabilities in natural language — mean
|err| 0.005 over textbook mechanisms, including non-textbook compounds like
"two biased coins 0.63 and 0.41, both heads" answered with 4-digit-exact
0.63×0.41 = **0.2583**. The *same model* expressing the *same beliefs* through
its token-selection channel is severely distorted: fair coin → 0.953/0.047
(TV 0.45), uniform K=20 lottery → TV 0.55, visible position/label bias, and
cross-primitive contradiction (Choice says heads 0.95 while Noul says
P(heads)=0.053 on the identical state). Five alternative elicitation methods
(answer-text scoring, permutation averaging, sampling frequencies, …) do not
fix it — only the verbalized channel is calibrated.

**2. The closed-source Jev API is miscalibrated too.** On a declared fair
coin, Jev's Choice returns P(heads) of **0.89–0.95** while its Noul returns
0.48–0.49 for the same event — a ~0.4 cross-primitive gap, even though the
model identifies the stated prior correctly 4/4 when asked as a fact. A
uniform K=255 lottery comes back as 0.75/0.84 per item (reference 1/255).
(Data: NanoJev's public probability-semantics probes plus our independent
receipts; our own 27B probe shows the same pattern.)

**3. Programmatic ground truth transfers calibration.** We generate 90k
probability questions (urns, lotteries, dice, coins, spinners, cards, dice
sums, plus deterministic sanity tasks) whose target distributions are *exact
analytic values* — no LLM in the loop. Distribution supervision (soft-label CE
+ Brier) on this data compresses Qwen3-0.6B's test TV from **0.471 → 0.082**
and its Choice-vs-Noul inconsistency from 0.754 → **0.024** — better
calibrated than Jev itself. On *unseen mechanism families* (cards, dice sums)
it still reaches Brier **0.019–0.108**: real generalization, not
interpolation.

**4. Specialists do not transfer.** Cross-checking both directions against
NanoJev: our probability specialist (v1) scores **11.2%** on the game cohort —
*below* the shared untuned base model (15.4%) — and NanoJev's public weights
score **38.5%** on our probability test — again below the shared base (49.7%).
The scoring-head architecture is no magic; generalization comes from data
coverage, not architecture.

**5. Mixed-domain training = a general System One.** v2 (same 0.6B
architecture, trained from scratch on a 1:1 mix of probability tasks and
NanoJev soft game decisions, 2000 steps) beats the base model in *both*
domains at once: probability acc **0.881** vs 0.497, game macro **48.8%** vs
15.4%. On games it is statistically indistinguishable from the Jev API
(paired McNemar p > 0.16 in all eight split×category cells) and beats NanoJev
on test Maze 6/10 vs 4/10 — all at **46–85 ms per single decision** on one
RTX 3090, with probability-domain calibration far beyond Jev.

## Architecture

```
 state + question + K candidates (dynamic set)
          │
          ▼
 chat template (single user message)
          │
          ▼
 K candidate paths:  [prompt] [label k] [EOS]     labels: 0-9, A-Z (single tokens)
          │
          ▼
 Qwen3-0.6B backbone  ── one packed forward (bf16 autocast, fp32 weights)
          │
          ▼
 EOS-position hidden state of each path
          │
          ▼
 LayerNorm → Linear (scalar score per path)
          │
          ▼
 group softmax over the K scores  ──►  full probability distribution (no text)
```

Training loss: soft-label cross-entropy + 0.1 × Brier against the exact
target distribution, computed per question after the group softmax.

## Quickstart

```bash
pip install -r requirements.txt
```

### Reproduce the data

```bash
# Probability tasks: 90k rows, exact analytic targets, seed-pinned (20260921)
python data/datagen.py --n 90000 --out data/train_v1.jsonl
pytest data/test_datagen.py        # 11 self-consistency tests (brute-force oracles)

# Game decisions: converted from NanoJev-Data (Jev teacher / visual-expert soft labels)
huggingface-cli download C-Tianyu/NanoJev-Data --repo-type dataset --local-dir /path/to/NanoJev-Data
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data
python data/make_gamedata.py       # -> data/games_v1.jsonl
```

100-row samples of both datasets are committed under `data/samples/` so the
format can be inspected without downloading anything.

### Train

```bash
# v1 — probability specialist (1200 steps, ~6 h on one RTX 3090)
python training/train.py --steps 1200 --out ckpt/v1

# v2 — general System One: 1:1 probability + game mix (2000 steps, ~10 h)
python training/train.py --steps 2000 --data data/train_v1.jsonl \
    --mix data/games_v1.jsonl --mix-ratio 0.5 \
    --out ckpt/v2 --log results/train_log_v2.jsonl
```

The backbone defaults to `Qwen/Qwen3-0.6B` (override with
`XIAOJEV_BASE_MODEL`); checkpoints are written as `backbone/` + `head.pt` +
`trainer.pt` + a sha256 manifest.

### Evaluate

```bash
python training/evaluate.py --ckpt ckpt/v2 --split test --output results/eval_v2_test.json
python training/evaluate.py --ckpt ckpt/v2 --split ood  --output results/eval_v2_ood.json
python training/evaluate.py --zero-shot --split test --output results/eval_zeroshot_test.json
```

Reports accuracy / NLL / Brier / ECE(10) / mean TV overall and per
mechanism×difficulty, plus paired Choice-vs-Noul consistency.

### Reproduce the zero-shot probes (finding 1)

Needs a local vLLM server; we used `cyankiwi/Qwen3.8-27B-AWQ-INT4`:

```bash
vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 --port 8020 --served-model-name qwen3.8-27b
export XIAOJEV_VLLM_URL=http://127.0.0.1:8020/v1   # default; XIAOJEV_VLLM_MODEL/XIAOJEV_MODEL_PATH too

python probes/probe_distributions.py   # token-channel distortion, cross-primitive gaps
python probes/probe_verbalized.py      # verbalized channel calibration
python probes/probe_novel.py           # novel (unmemorizable) mechanisms
python probes/probe_elicitation.py     # five elicitation methods compared
```

### Reproduce the comparisons (findings 4–5)

```bash
export XIAOJEV_NANOJEV_REPO=/path/to/NanoJev        # clone of github.com/TianyuCodings/NanoJev
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data

python comparison/compare_sanity.py      # re-verify NanoJev's published 548-case numbers
python comparison/compare_rollout.py --engine vcdm --checkpoint ckpt/v2 \
    --output results/compare_v2_frozen_episodes.jsonl
python comparison/compare_report.py results/compare_v2_frozen_episodes.jsonl \
    results/compare_v2_frozen.json
python comparison/compare_reverse.py     # NanoJev weights on our probability test
python comparison/latency_bench.py       # v1 / v2 / NanoJev latency (--skip-nanojev to skip)
```

(`vcdm` is the internal engine codename for the xiaojev checkpoints, kept for
artifact compatibility. Game rollouts of the Doom scenarios additionally need
`vizdoom`.)

## Full results

Every number cited above, with per-cell tables and the artifact file each
table is computed from: **[docs/RESULTS.md](docs/RESULTS.md)**. Summary JSONs
live in `results/`; per-row predictions and trajectories are reproducible but
not committed.

## Checkpoints

Two checkpoints (~14 GB total, including optimizer states):

- `xiaojev-v1` — probability specialist (TV 0.082)
- `xiaojev-v2` — general System One (probability + games)

> **TODO:** upload to HuggingFace — link placeholder `https://huggingface.co/TODO/xiaojev`.
> Until then, both can be retrained from scratch with the commands above;
> data generation is seed-pinned and the training seed defaults to 0.

## Requirements

- One 24 GB GPU (developed on a single RTX 3090)
- Python 3.12+ (developed on 3.10 with torch 2.10.0+cu128, transformers 4.57.6)
- `torch` 2.10, `transformers` 4.57, `httpx`, `pytest`; `vizdoom` only for the
  Doom game rollouts; vLLM only for serving the 27B probe target

## Limitations

- **Research prototype.** Two checkpoints, one GPU, one random seed; no
  extensive hyperparameter search.
- **8K context.** States are truncated to fit an 8192-token budget.
- **Domain coverage.** Trained on programmatic probability mechanisms and four
  game tasks (Maze, Snake, Doom basic, Doom predict_position). Semantic
  decision domains — browser-use, RAG, business choices — are *not* trained
  or evaluated; calibration there is unknown.
- **Game supervision is teacher distillation** (Jev native_probs / visual
  expert policy), not ground truth; only the probability domain has exact
  analytic targets.
- **No affiliation with TypeSafe.** "Jev" is referenced solely as a benchmark
  via NanoJev's published artifacts and public API receipts; xiaojev is an
  independent re-implementation study.

## Attribution

- [NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT) — game training
  data (NanoJev-Data), the frozen 548-case comparison protocol and verifier,
  and the Jev API receipts we benchmark against.
- [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT) —
  browser-use domain reference point for System One-style decision models.
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0) — base
  backbone (0.6B); the 27B probe target is an AWQ community quant of Qwen3.8-27B.

## License

MIT — see [LICENSE](LICENSE). Datasets and checkpoints derived from NanoJev
and Qwen3 remain under their respective licenses.
