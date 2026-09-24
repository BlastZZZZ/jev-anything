"""Pure-python BM25 over the musique corpus (11,656 docs).

Outputs:
  bm25_inpool.jsonl  - per-question ranking of that question's own 20 paragraphs
  bm25_corpus.jsonl  - per-question top-50 docids from the full corpus (+ gold docids)
IDF is corpus-level in both cases. k1=1.5, b=0.75. No new packages.
"""

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from rag_eval.common import load_corpus, load_questions, tokens

OUT = Path(os.environ.get("XIAOJEV_RETRIEVAL_DIR", "results/retrieval"))
K1, B = 1.5, 0.75


def norm_text(t):
    return re.sub(r"\s+", " ", t).strip()


def build_index(corpus):
    """Inverted index + idf over the corpus. Returns (postings, idf, doc_len, avgdl)."""
    df = defaultdict(int)
    postings = defaultdict(list)
    doc_len = []
    for di, c in enumerate(corpus):
        toks = tokens(c["title"] + " " + c["text"])
        doc_len.append(len(toks))
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, f in tf.items():
            df[t] += 1
            postings[t].append((di, f))
    N = len(corpus)
    avgdl = sum(doc_len) / N
    idf = {t: math.log(1 + (N - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    return postings, idf, doc_len, avgdl


def score_query(qtoks, postings, idf, doc_len, avgdl):
    scores = defaultdict(float)
    for t in set(qtoks):
        if t not in idf:
            continue
        w = idf[t]
        for di, f in postings[t]:
            scores[di] += (
                w * (f * (K1 + 1)) / (f + K1 * (1 - B + B * doc_len[di] / avgdl))
            )
    return scores


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    corpus = load_corpus()
    questions = load_questions()
    print(f"corpus {len(corpus)} docs, {len(questions)} questions")

    # map question paragraphs to corpus docids by normalized (title, text)
    key2doc = {}
    for c in corpus:
        key2doc[(c["title"], norm_text(c["text"]))] = c["docid"]
    para_docid = {}
    misses = 0
    for q in questions:
        for idx, title, text in q["paragraphs"]:
            d = key2doc.get((title, norm_text(text)))
            if d is None:
                misses += 1
            para_docid[(q["id"], idx)] = d
    print(f"paragraph->corpus mapping misses: {misses}")

    # build inverted index
    df = defaultdict(int)
    postings = defaultdict(list)  # term -> [(doc_int, tf)]
    doc_len = []
    for di, c in enumerate(corpus):
        toks = tokens(c["title"] + " " + c["text"])
        doc_len.append(len(toks))
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, f in tf.items():
            df[t] += 1
            postings[t].append((di, f))
    N = len(corpus)
    avgdl = sum(doc_len) / N
    idf = {t: math.log(1 + (N - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    print(f"index built in {time.time() - t0:.1f}s, vocab {len(df)}, avgdl {avgdl:.1f}")

    def score_query(qtoks):
        scores = defaultdict(float)
        for t in set(qtoks):
            if t not in idf:
                continue
            w = idf[t]
            for di, f in postings[t]:
                scores[di] += (
                    w * (f * (K1 + 1)) / (f + K1 * (1 - B + B * doc_len[di] / avgdl))
                )
        return scores

    with (
        open(OUT / "bm25_inpool.jsonl", "w") as fi,
        open(OUT / "bm25_corpus.jsonl", "w") as fc,
    ):
        for qi, q in enumerate(questions):
            qtoks = tokens(q["question"])
            # in-pool: rank own paragraphs using corpus IDF
            pool_scores = {}
            for idx, title, text in q["paragraphs"]:
                tf = defaultdict(int)
                p_toks = tokens(title + " " + text)
                for t in p_toks:
                    tf[t] += 1
                dl = len(p_toks)
                s = 0.0
                for t in set(qtoks):
                    if t in idf and t in tf:
                        f = tf[t]
                        s += (
                            idf[t]
                            * (f * (K1 + 1))
                            / (f + K1 * (1 - B + B * dl / avgdl))
                        )
                pool_scores[idx] = s
            ranking = sorted(pool_scores, key=lambda i: -pool_scores[i])
            fi.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "hop": q["hop"],
                        "split": q["split"],
                        "gold_idx": q["gold_idx"],
                        "ranking": ranking,
                        "scores": {str(k): round(v, 4) for k, v in pool_scores.items()},
                    }
                )
                + "\n"
            )
            # corpus-level
            sc = score_query(qtoks)
            top = sorted(sc, key=lambda d: -sc[d])[:50]
            fc.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "hop": q["hop"],
                        "split": q["split"],
                        "gold_docids": [
                            para_docid[(q["id"], i)] for i in q["gold_idx"]
                        ],
                        "top50": [corpus[d]["docid"] for d in top],
                        "top50_titles": [corpus[d]["title"] for d in top],
                    }
                )
                + "\n"
            )
            if (qi + 1) % 200 == 0:
                print(f"{qi + 1}/{len(questions)} questions, {time.time() - t0:.1f}s")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
