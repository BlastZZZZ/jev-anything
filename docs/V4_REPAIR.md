# v4 browser and dense-reranking repair

This release preserves the original five-source v4 checkpoint and adds a separate browser adaptation plus a RAG rank-fusion layer. Code, frozen metrics, document-level ranking inputs, and local fixture traces are included. Model binaries and full training datasets are not hosted in this Git repository; the checkpoint integrity manifests and reproduction instructions identify the evaluated artifacts.

## Browser

| Local fixture regression | Actions | Decision calls | Text calls | Actual final page |
|---|---:|---:|---:|---|
| Lisbon / Design / Free cancellation / Casa Flora, repeat 1 | 5 | 6 | 1 | Verified |
| Same original task, repeat 2 | 5 | 6 | 1 | Verified |
| Copenhagen / Design / Free cancellation / The Glasshouse | 5 | 6 | 1 | Verified |
| Lisbon / Nature / Free cancellation off / Serra Lodge | 4 | 5 | 1 | Verified |

Original v4 passed 0/2, repeating Find stays 15 times per run. The repair retains empty field values, checked/selected/expanded states, dropdown options, and recent input values. Training and runtime share the request/row serialization. The progress check compares page content and semantic action fields rather than transient node IDs, so equivalent DOM reconstruction cannot keep a repeated-click loop alive. Three non-WAIT actions with no observable progress stop the agent.

The first adaptation stage uses 16,547 programmatically labeled mixed-control rows. The second uses 900 actually rendered counterfactual DOM states, producing 5,007 rows (4,179 train / 828 dev), with scenario-disjoint splits and consistent index permutations. Only the final six backbone layers, normalization layers, and scoring head are trained.

The second-stage step100 checkpoint achieved 82% on 150 fixed dev rows and passed 4/4 real regressions twice. Step150 reached 86% dev accuracy but passed only 3/4, omitting the Copenhagen search submission; it was rejected. Deployment selection therefore uses the browser regressions as well as dev accuracy. The final artifact is `ckpt/v4_browser_dom/step100`, with the local alias `ckpt/v4_browser`.

Evidence: [final summary](../results/v4_repair/browser_final_summary.json), [first passing run](../results/v4_repair/browser_dom100_summary.json), [rejected step150](../results/v4_repair/browser_dom150_summary.json), [selection](../results/v4_repair/browser_selected.json), and the corresponding full JSON traces in `results/v4_repair`. The acceptance script checks completion status, actual detail URL, and rendered filter text; `DONE` alone never counts as success. The local 27B helper writes field text only.

**Limits:** These are three distinct tasks, including one repeated task, on a local fixture whose layout is present in adaptation data. Cases are used for deployment selection, so this is not an independent general-web benchmark. Current adaptation goals exclude the evaluated city/hotel names, but the original v4 had seen those entities. There are no site-specific runtime plans or fixed field values. Other domains have not been re-evaluated with browser-adapted weights.

## Dense retrieval and RAG

The original pipeline discarded dense ordering and sorted solely by v4's relevance score. The repair fuses both ranks:

`score(document) = 0.6 / (1 + dense_rank) + 0.4 / (1 + v4_rank)`

Ranks start at 1. Ties preserve dense order. We selected weights and rank constant using only 98 calibration questions, froze them, and then evaluated dev/test. The original v4 weights remain unchanged; fused ranks are not probabilities.

| Split | Questions | Dense R@5 | Original v4 reorder R@5 | Fusion R@5 |
|---|---:|---:|---:|---:|
| Independent test | 101 | 73.35% | 65.35% | **77.31%** |
| Dev | 94 | 68.88% | — | **74.11%** |
| All non-training, including calibration | 293 | 70.68% | 66.10% | **76.14%** |

On independent test, the gain over dense is 3.96 percentage points, with paired-bootstrap 95% interval +0.91 to +7.01 points; 17 questions improve and 5 worsen. Frozen [parameters](../results/v4_repair/fusion_config.json), [metrics](../results/v4_repair/fusion_metrics.json), [calibration trials](../results/v4_repair/fusion_calibration_trials.json), and [per-question rankings/scores](../results/v4_repair/retrieval_inputs.jsonl) are included. The offline evaluator reproduces these results without model downloads or a corpus.

With the same Qwen reader and top-4 context, all-293 EM rises from dense's 31.06% to 34.13% (original v4: 28.33%), and F1 from 41.04% to 44.75%. Independent-test EM rises from 35.64% to 36.63%, and F1 from 44.88% to 46.60%. These small QA gains are not claimed statistically significant. Dense reader baselines are retained same-configuration runs; all 293 new reader requests succeeded. See [QA metrics](../results/v4_repair/qa_fusion_metrics.json).

The live `DenseReranker` accepts any dense-ordered candidate dictionaries and recomputes v4 scores before fusion, with no gold labels or cached scores at runtime. A three-query calibration smoke reproduced the frozen top-five orders exactly. See [usage and offline reproduction](../rag_eval/README.md).

## Original v4 across domains

The original v4 mixes probability, games, semantic decisions, browser decisions, and hard-negative RAG decisions at equal source weights for 2,500 steps. It has different tradeoffs from v3; this browser/RAG repair does not erase those results.

| Metric | v3 | Original v4 |
|---|---:|---:|
| Game weighted macro success, test | 42.16% | 53.26% |
| Game weighted macro success, OOD | 15.76% | 26.72% |
| Probability test accuracy | 87.97% | 85.62% |
| Probability test TV, lower is better | 0.1045 | 0.1264 |
| Semantic test accuracy | 85.99% | 83.52% |

All 548 frozen game episodes and 21,745 state transitions completed verification in the original acceptance run. Game OOD remains slightly below v2's 28.72%. Frozen per-domain summaries are in [results/v4](../results/v4). Original v4 single-question inference p50 was 27.84–69.49 ms across domains on one RTX 3090 after warmup; this excludes browser execution and RAG reader latency.

## Reproduction and validation

Generate the original probability/game/semantic sources as documented in the main README, plus `python data/make_browserdata.py` and the [RAG data pipeline](../rag_eval/README.md). Then train original v4:

```bash
python training/train.py --steps 2500 \
  --mix data/train_v1.jsonl:0.2 --mix data/games_v1.jsonl:0.2 \
  --mix data/semantic_v1.jsonl:0.2 --mix data/browser_v1.jsonl:0.2 \
  --mix data/rag_v1.jsonl:0.2 --out ckpt/v4 --log results/train_log_v4.jsonl
```

Browser adaptation commands and checkpoint selection are documented in [experiments/v4_browser](../experiments/v4_browser/README.md). Use the original v4 for RAG and the separate browser checkpoint for the local agent. Exact dense-negative reproduction also needs the original dense index/cache; BM25 fallback is a different training-data configuration.

The local repair passed 37 browser pytest checks including CUDA inference, two fusion tests, JavaScript syntax checks, Ruff, and the upstream package build. Checkpoint SHA256 hashes were verified against their manifests. See [checkpoint integrity](../results/v4_repair/checkpoint_integrity.json). Published paths are repository-relative or configurable via environment variables; full datasets and weights remain outside Git.

The portable GitHub export was also checked separately: 13 core/data tests passed (five semantic-data tests skipped because the full source datasets are not included), all 30,800 generated browser-data records passed their invariants, and all reported fusion metrics/bootstrap intervals reproduced within 1e-12 across Python versions. The published live reranker matched frozen top-five results on three calibration queries. Installing the patch into a clean copy of the pinned browser agent passed 37 tests including CUDA inference, Ruff, both JavaScript syntax checks, and wheel/sdist builds. Reapplying the installer was verified to be idempotent.
