# Dense retrieval with xiaojev v4

The repair combines the dense retriever's order with the original v4 model's order. It uses weighted reciprocal-rank fusion, with dense weight 0.6, reranker weight 0.4, and rank constant 1. Parameters were selected on 98 calibration questions and frozen before evaluating dev/test.

```python
from rag_eval.dense_reranker import DenseReranker

reranker = DenseReranker(
    checkpoint="ckpt/v4",
    config_path="results/v4_repair/fusion_config.json",
)
# Passages are dictionaries with docid, title, text, in descending dense-score order.
ranked = reranker.rerank(question, passages[:50])
```

This interface scores the supplied candidates on CUDA; it requires the original v4 weights and the base tokenizer (`XIAOJEV_BASE_MODEL`, default `Qwen/Qwen3-0.6B`). It does not load gold labels or the frozen evaluation inputs. The returned `xiaojev_p_relevant` is the original model score; the fused ranking is not a calibrated probability.

## Recompute the reported metrics

From the repository root, using only Python's standard library:

```bash
python -m rag_eval.evaluate_fusion --verify-calibration --output results/recomputed_fusion.json
```

The committed `results/v4_repair/retrieval_inputs.jsonl` contains document IDs, frozen dense rankings, original v4 relevance scores, gold document IDs, and split labels for all 293 non-training questions. It contains no corpus text. This command reproduces the calibration choice, split metrics, and paired bootstrap interval; it does not rerun the embedding model or reader.

Independent test (101 questions): dense R@5 73.35%, original v4-only reordering 65.35%, fusion 77.31%. The paired bootstrap 95% interval for the gain over dense is +0.91 to +7.01 percentage points. The 293-question total includes calibration and must not be called an independent test result.

## Data preparation for the original v4 training

Set `XIAOJEV_RAG_DATA` to a directory with `musique/raw/musique.json`, `musique/gold.jsonl`, and `musique/corpus.jsonl`, using the normalized dataset format described in the main README.

```bash
python -m rag_eval.bm25_rank
python data/make_ragdata.py
python scripts/check_rag_data.py
```

BM25 outputs default to `results/retrieval`; override with `XIAOJEV_RETRIEVAL_DIR`. Dense hard negatives can reuse an NV-Embed-v2 index: `XIAOJEV_DENSE_INDEX_ROOT` must contain `index/inputs.json` with a `chunk` list of `{id, content}` records and `index/chunk_vectors.npy` in the same row order. The embedding endpoint is set with `XIAOJEV_EMBED_URL` (default `http://127.0.0.1:8019/v1/embeddings`). Without that index/service, the generator explicitly reports its BM25 fallback; that fallback does not exactly reproduce the original dense-negative training run.

QA scores and the small test-set QA improvement are recorded in [the release report](../docs/V4_REPAIR.md). Reader baselines are retained same-configuration runs, not fresh paired reruns.
