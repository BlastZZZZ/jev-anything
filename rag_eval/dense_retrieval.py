"""Dense first-stage retrieval: NV-Embed-v2 query embeddings x reused corpus
chunk vectors (from the HippoRAGv2 run's index, exact-text reuse, 4096-d).

Writes dense_corpus.jsonl {id, top50, gold_docids} for non-train questions,
plus dense-alone retrieval metrics printed to stdout.
"""

import json
import os
import re
from pathlib import Path
from urllib import request

import numpy as np

from rag_eval.common import load_corpus, load_questions

OUT = Path(os.environ.get("XIAOJEV_RETRIEVAL_DIR", "results/retrieval"))
HIPPO = Path(os.environ.get("XIAOJEV_DENSE_INDEX_ROOT", "datasets/dense_index"))
EMB_API = os.environ.get("XIAOJEV_EMBED_URL", "http://127.0.0.1:8019/v1/embeddings")
QUERY_PREFIX = (
    "Instruct: Given a question, retrieve relevant documents that "
    "best answer the question.\nQuery: "
)
NONTRAIN = ("dev", "calibration", "test")


def norm_text(t):
    return re.sub(r"\s+", " ", t).strip()


def embed(texts, batch=16):
    vecs = []
    for i in range(0, len(texts), batch):
        payload = {
            "model": "nvidia/NV-Embed-v2",
            "input": texts[i : i + batch],
            "encoding_format": "float",
        }
        req = request.Request(
            EMB_API,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read())
        vecs.extend(e["embedding"] for e in sorted(d["data"], key=lambda x: x["index"]))
    return np.asarray(vecs, dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    questions = [q for q in load_questions() if q["split"] in NONTRAIN]
    key2doc = {(c["title"], norm_text(c["text"])): c["docid"] for c in corpus}

    inputs = json.load(open(HIPPO / "index" / "inputs.json"))["chunk"]
    # Preserve inputs order and verify its correspondence to vector rows below.
    vecs = np.load(HIPPO / "index" / "chunk_vectors.npy", mmap_mode="r")
    assert vecs.shape[0] == len(inputs) == 11656

    # map rows to docids by content "title\ntext"
    row_docid = []
    misses = 0
    for c in inputs:
        title, _, text = c["content"].partition("\n")
        d = key2doc.get((title, norm_text(text)))
        if d is None:
            misses += 1
        row_docid.append(d)
    print(f"chunk->docid misses: {misses}")
    assert misses == 0

    # verify ordering: encode one known doc and compare to its row
    probe_idx = 0
    probe_vec = embed([inputs[probe_idx]["content"]])[0]
    row = np.asarray(vecs[probe_idx])
    cos = float(probe_vec @ row / (np.linalg.norm(probe_vec) * np.linalg.norm(row)))
    print(f"order probe cos(row0, re-encode) = {cos:.4f}")
    assert cos > 0.999, "chunk_vectors row order does not match inputs.json order"

    qs = [QUERY_PREFIX + q["question"] for q in questions]
    qv = embed(qs)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    dv = np.array(vecs, dtype=np.float32)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)

    ks = (1, 5, 10, 20, 50)
    agg = {f"r@{k}": 0.0 for k in ks}
    agg.update({f"all@{k}": 0.0 for k in ks})
    with open(OUT / "dense_corpus.jsonl", "w") as f:
        for qi, q in enumerate(questions):
            sims = qv[qi] @ dv.T
            top = np.argpartition(-sims, 50)[:50]
            top = top[np.argsort(-sims[top])]
            top_docids = [row_docid[i] for i in top]
            gold = set()
            for idx in q["gold_idx"]:
                t, x = next((t, x) for i, t, x in q["paragraphs"] if i == idx)
                gold.add(key2doc[(t, norm_text(x))])
            f.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "hop": q["hop"],
                        "gold_docids": sorted(gold),
                        "top50": top_docids,
                    }
                )
                + "\n"
            )
            for k in ks:
                agg[f"r@{k}"] += len(set(top_docids[:k]) & gold) / len(gold)
                agg[f"all@{k}"] += float(gold <= set(top_docids[:k]))
    n = len(questions)
    print(f"dense retrieval (NV-Embed-v2, non-train n={n}):")
    for k in ks:
        print(f"  r@{k}={agg[f'r@{k}'] / n:.3f} all@{k}={agg[f'all@{k}'] / n:.3f}")


if __name__ == "__main__":
    main()
