# jev anything

**Lightweight 0.6B decision models for browser agents, RAG, games, and probabilistic reasoning.**

jev anything scores a dynamic set of candidates and returns a probability distribution
that applications can use to choose an action, rank evidence, or assess uncertainty.
It runs locally on a single GPU and selects among supplied candidates without
autoregressive text generation.

- **Browser actions:** choose operations and observed element targets for search, form filling, filtering, and navigation.
- **RAG retrieval:** score passage relevance and combine reranker and dense-retriever rankings.
- **Evidence assessment:** select supporting passages and judge answerability and context sufficiency.
- **Game policies:** select actions in Maze, Snake, and Doom environments.
- **Probabilistic reasoning:** predict outcome distributions for coins, dice, urns, lotteries, and related decision tasks.

[中文文档](README.zh-CN.md) · [Evaluation details](docs/V4_REPAIR.md) · [Research results](docs/RESULTS.md)

## Results

| Capability | Evaluation | Result |
|---|---|---:|
| Browser actions | Local hotel tasks, independently checked final pages | **4/4** |
| RAG retrieval | MuSiQue test R@5, 101 questions, dense + jev anything fusion | **77.31%** |
| RAG pipeline | MuSiQue test QA EM / F1, 101 questions, top-4 context | **36.63% / 46.60%** |
| Evidence assessment | Semantic test accuracy, 2,384 decisions | **83.52%** |
| Game policies | Weighted macro success, test / OOD | **53.26% / 26.72%** |
| Probabilistic reasoning | Probability test accuracy / mean TV, 8,145 decisions | **85.62% / 0.1264** |
| Inference | Single-decision p50 across domains, one RTX 3090 | **27.84–69.49 ms** |

Browser results use the separately adapted browser checkpoint and cover three
local fixture tasks, including one repeated task. The fixture layout is present
in adaptation data, and these cases participate in checkpoint selection.
RAG retrieval uses the v4 checkpoint with rank fusion; QA answers come from a
separate Qwen reader. Other model metrics use v4. Lower TV means better
probability calibration. Latency measures warmed-up model inference, including
tokenization, and excludes browser execution and the QA reader.

Reproduce the RAG ranking results offline, without model downloads:

```bash
python -m rag_eval.evaluate_fusion --verify-calibration
```

[Live RAG interface](rag_eval/README.md) ·
[Browser installation](integrations/jev-ultrafast/README.md) ·
[Browser training](experiments/v4_browser/README.md)

## Research background

Token probabilities from a general language model can differ substantially from
its verbalized probability estimates. jev anything uses a dedicated scoring head and
distribution supervision to learn a decision interface. Probability tasks use
analytic targets, semantic and RAG tasks use QA gold labels, browser tasks use
programmatic interaction labels, and game tasks use teacher or expert policies.
Probe results, calibration measurements, and training studies are recorded in
[the research report](docs/RESULTS.md).

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

Training loss: soft-label cross-entropy + 0.1 × Brier against the
target distribution, computed per question after the group softmax.

## Quickstart

```bash
git clone https://github.com/BlastZZZZ/jev-anything.git
cd jev-anything
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

# Semantic decisions: 24k rows from gold supporting facts of
# HotpotQA / 2WikiMultiHopQA / MuSiQue — zero LLM calls, seed-pinned (20260922)
export XIAOJEV_RAG_DATA=/path/to/rag_datasets   # hotpotqa/ 2wikimultihopqa/ musique/,
                                                # each with raw/<name>.json + gold.jsonl + corpus.jsonl
python data/make_semanticdata.py   # -> data/semantic_v1.jsonl
pytest data/test_semanticdata.py   # truth mapping, split leakage, class balance, negatives
```

Generate browser decisions with `python data/make_browserdata.py`. Prepare the
hard-negative RAG source with the [RAG data pipeline](rag_eval/README.md).
100-row samples of the training sources and browser adaptation data are
committed under `data/samples/` so the formats can be inspected directly.

### Train

```bash
# Five-source training: probability, games, semantics, browser, and RAG
python training/train.py --steps 2500 \
    --mix data/train_v1.jsonl:0.2 --mix data/games_v1.jsonl:0.2 \
    --mix data/semantic_v1.jsonl:0.2 --mix data/browser_v1.jsonl:0.2 \
    --mix data/rag_v1.jsonl:0.2 \
    --out ckpt/v4 --log results/train_log_v4.jsonl
```

The backbone defaults to `Qwen/Qwen3-0.6B` (override with
`XIAOJEV_BASE_MODEL`); checkpoints are written as `backbone/` + `head.pt` +
`trainer.pt` + a sha256 manifest.

### Evaluate

```bash
python training/evaluate.py --ckpt ckpt/v4 --split test --output results/eval_v4_test.json
python training/evaluate.py --ckpt ckpt/v4 --split ood  --output results/eval_v4_ood.json
python training/evaluate.py --ckpt ckpt/v4 --split test --data data/semantic_v1.jsonl \
    --output results/eval_v4_semantic_test.json
python training/evaluate.py --zero-shot --split test --data data/semantic_v1.jsonl \
    --output results/eval_zs_semantic_test.json
```

Reports accuracy / NLL / Brier / ECE(10) / mean TV overall and per
mechanism×difficulty, plus paired Choice-vs-Noul consistency.

### Reproduce the zero-shot probes

Needs a local vLLM server; we used `cyankiwi/Qwen3.8-27B-AWQ-INT4`:

```bash
vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 --port 8020 --served-model-name qwen3.8-27b
export XIAOJEV_VLLM_URL=http://127.0.0.1:8020/v1   # default; XIAOJEV_VLLM_MODEL/XIAOJEV_MODEL_PATH too

python probes/probe_distributions.py   # token-channel distortion, cross-primitive gaps
python probes/probe_verbalized.py      # verbalized channel calibration
python probes/probe_novel.py           # novel (unmemorizable) mechanisms
python probes/probe_elicitation.py     # five elicitation methods compared
```

### Evaluate game policies

```bash
export XIAOJEV_NANOJEV_REPO=/path/to/NanoJev        # clone of github.com/TianyuCodings/NanoJev
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data

python comparison/compare_sanity.py      # re-verify NanoJev's published 548-case numbers
python comparison/compare_rollout.py --engine vcdm --checkpoint ckpt/v4 \
    --output results/compare_v4_frozen_episodes.jsonl
python comparison/compare_report.py results/compare_v4_frozen_episodes.jsonl \
    results/compare_v4_frozen.json
```

(`vcdm` is the internal engine codename for the jev anything checkpoints, kept for
artifact compatibility. Game rollouts of the Doom scenarios additionally need
`vizdoom`.)

## Browser agent integration

[`integrations/jev-ultrafast/`](integrations/jev-ultrafast/) includes the local
backend, a patch against a pinned upstream commit, and an installer. It preserves
control state and detects repeated actions without observable progress. Set
`JEV_BACKEND=local`; the local scoring model selects operations/targets and an
OpenAI-compatible text helper supplies only `TYPE_TEXT` values. Candidate
scoring uses bounded microbatches with zero autoregressive decode steps.

The browser checkpoint passed four local hotel regressions twice,
using 5, 5, 5, and 4 actions. Actual URLs and filter text were checked. See the
[installation and limitations](integrations/jev-ultrafast/README.md) and
[final evidence](results/v4_repair/browser_final_summary.json).

## Full results

Current v4 metrics, checkpoint selection, validation limits, and evidence links
are in **[docs/V4_REPAIR.md](docs/V4_REPAIR.md)**. Historical v1–v3 tables remain
in [docs/RESULTS.md](docs/RESULTS.md). Summary JSONs live in `results/`; local
fixture traces and frozen document-ranking inputs are included.

## Checkpoints

- `ckpt/v4`: probability, games, semantic decisions, browser decisions, and hard-negative RAG training.
- `ckpt/v4_browser_dom/step100`: browser-adapted weights; `ckpt/v4_browser` is its local alias.

Model binaries are not hosted in this Git repository and a public weight
download is not yet available. The release includes checkpoint hashes,
training/data code, and frozen evaluation evidence. Model training and
browser adaptation commands are in the [evaluation report](docs/V4_REPAIR.md).
Use the browser checkpoint for the agent and v4 for the other evaluated domains.

## Requirements

- One 24 GB GPU (developed on a single RTX 3090)
- Python 3.10+ for the core; Python 3.12+ for the browser integration
- `torch` 2.10, `transformers` 4.57, `httpx`, `pytest`; `vizdoom` only for the
  Doom game rollouts; vLLM only for serving the 27B probe target

## Limitations

- **Research prototype.** Small-scale checkpoint experiments, one GPU, one training seed; no
  extensive hyperparameter search.
- **8K context.** States are truncated to fit an 8192-token budget.
- **Domain coverage.** Trained on programmatic probability mechanisms, four
  game tasks (Maze, Snake, Doom basic, Doom predict_position), and four
  RAG-style semantic decisions built from QA gold labels, with browser and
  hard-negative RAG sources added in v4. Browser adaptation is validated only
  on a local fixture; arbitrary websites remain untested.
- **Game supervision is teacher distillation** (Jev native_probs / visual
  expert policy), not ground truth; probability and semantic domains have
  exact targets (analytic / gold-label-derived).
- **No affiliation with TypeSafe.** "Jev" is referenced solely as a benchmark
  via NanoJev's published artifacts and public API receipts; jev anything is an
  independent re-implementation study.

## Attribution

- [NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT) — game training
  data (NanoJev-Data), the frozen 548-case comparison protocol and verifier,
  and the Jev API receipts we benchmark against.
- [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT) —
  browser-use agent that jev anything integrates with as a local decision backend
  (`integrations/jev-ultrafast/`).
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0) — base
  backbone (0.6B); the 27B probe target is an AWQ community quant of Qwen3.8-27B.
- HotpotQA, 2WikiMultiHopQA, MuSiQue — source QA datasets whose gold
  supporting facts supervise the semantic domain (each under its own license).

## License

MIT — see [LICENSE](LICENSE). Datasets and checkpoints derived from NanoJev
and Qwen3 remain under their respective licenses.
