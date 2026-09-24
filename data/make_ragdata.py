"""Build rag_v1: hard-negative RAG decision data from musique TRAIN-split
questions only (the 293 non-train questions used for v3 evaluation are strictly
excluded; disjointness is asserted here and re-verified in test_ragdata.py).

Motivation (rag_eval findings, 2026-09-23): v3's P(yes) saturates on strong
candidates (~20% of dense top-50 > 0.99) because semantic_v1 trained relevance
only against filtered random-corpus negatives. rag_v1 mines HARD negatives:
same-question distractors and corpus-level BM25/dense top-50 non-gold docs.

Two item types, schema compatible with train.py load_rows/encode_row:
  rag_passage_relevance  noul: question + single passage -> yes/no
  rag_evidence_choice    choice K=5-6: 1-2 gold + 3-4 hard negatives

Hard negatives are >= 50% of all negatives (asserted). Splits are by question
id within the musique train set (70/10/10/10 train/dev/calibration/test,
second-level hash, seed 20260923). Prompt formats identical to
make_semanticdata.py (en/zh per-question by hash parity).

Dense negatives need the local NV-Embed-v2 service (127.0.0.1:8019); if it is
unavailable, dense negatives fall back to extra corpus-BM25 negatives and the
hard-ratio invariant is preserved (dense cache: data/rag_v1_dense_top50.jsonl).

Run: python3 make_ragdata.py
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_eval.common import load_questions, load_corpus, passage_block  # noqa
from make_semanticdata import split_of as sem_split_of, random_corpus_negatives  # noqa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rag_v1.jsonl"
SPLITS_OUT = ROOT / "data" / "rag_v1.splits.json"
DENSE_CACHE = ROOT / "data" / "rag_v1_dense_top50.jsonl"
RAG_EVAL = Path(os.environ.get("XIAOJEV_RETRIEVAL_DIR", "results/retrieval"))
SEED = 20260923

INSTR_CHOICE = {
    "en": "Which of the following passages contains the information needed to answer the question above?",
    "zh": "以上哪条段落包含回答该问题所需的信息？",
}
PROP_RELEVANCE = {
    "en": "The passage above helps answer the question.",
    "zh": "以上段落有助于回答所提的问题。",
}


def rag_split_of(qid):
    """Second-level split within musique train questions."""
    h = int(hashlib.sha256(f"rag:{qid}:{SEED}".encode()).hexdigest()[:8], 16) % 100
    return (
        "train" if h < 70 else "dev" if h < 80 else "calibration" if h < 90 else "test"
    )


def lang_of(qid):
    return (
        "en" if int(hashlib.sha256(qid.encode()).hexdigest()[:4], 16) % 2 == 0 else "zh"
    )


class Corpus:
    def __init__(self):
        self.docs = load_corpus()
        self.by_docid = {c["docid"]: c for c in self.docs}
        self.title_text = [(c["title"], c["text"]) for c in self.docs]


def make_item(
    qid,
    kind,
    variant,
    primitive,
    state,
    instruction,
    proposition,
    candidates,
    target,
    meta,
):
    return {
        "id": f"rag_{qid}_{kind}_{variant}",
        "family": "rag",
        "mechanism": f"rag_{kind}",
        "primitive": primitive,
        "state": state,
        "instruction": instruction,
        "proposition": proposition,
        "candidates": candidates,
        "target_distribution": target,
        "difficulty": "simple",
        "split": rag_split_of(qid),
        "paired_id": None,
        "meta": {
            "dataset": "musique",
            "question_id": qid,
            "lang": lang_of(qid),
            **meta,
        },
    }


def dense_top50(train_qids):
    """NV-Embed-v2 top-50 docids per train question, cached to JSONL."""
    if DENSE_CACHE.exists():
        return {d["id"]: d["top50"] for d in map(json.loads, open(DENSE_CACHE))}
    try:
        import numpy as np

        from rag_eval.dense_retrieval import HIPPO, QUERY_PREFIX, embed, norm_text
    except Exception as e:
        print(
            f"dense unavailable (import): {e!r}; falling back to BM25-only hard negatives"
        )
        return {}
    try:
        questions = {q["id"]: q for q in load_questions()}
        inputs = json.load(open(HIPPO / "index" / "inputs.json"))["chunk"]
        corpus = Corpus()
        key2doc = {
            (t, norm_text(x)): d
            for d, (t, x) in (
                (c["docid"], (c["title"], c["text"])) for c in corpus.docs
            )
        }
        row_docid = []
        for c in inputs:
            title, _, text = c["content"].partition("\n")
            row_docid.append(key2doc.get((title, norm_text(text))))
        assert all(row_docid)
        vecs = np.array(
            np.load(HIPPO / "index" / "chunk_vectors.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        qlist = sorted(train_qids)
        qv = embed([QUERY_PREFIX + questions[q]["question"] for q in qlist])
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        out = {}
        for i, qid in enumerate(qlist):
            sims = qv[i] @ vecs.T
            top = np.argpartition(-sims, 50)[:50]
            top = top[np.argsort(-sims[top])]
            out[qid] = [row_docid[j] for j in top]
        with open(DENSE_CACHE, "w") as f:
            for qid in qlist:
                f.write(json.dumps({"id": qid, "top50": out[qid]}) + "\n")
        print(
            f"dense top-50 embedded for {len(qlist)} train questions -> {DENSE_CACHE}"
        )
        return out
    except Exception as e:
        print(
            f"dense unavailable (encode): {e!r}; falling back to BM25-only hard negatives"
        )
        return {}


def main():
    rng = random.Random(SEED)
    corpus = Corpus()
    questions = load_questions()

    train_qs = [q for q in questions if sem_split_of("musique", q["id"]) == "train"]
    held_out = {
        q["id"] for q in questions if sem_split_of("musique", q["id"]) != "train"
    }
    assert len(train_qs) == 707 and not {q["id"] for q in train_qs} & held_out
    print(
        f"train-split questions: {len(train_qs)}; held-out eval questions excluded: {len(held_out)}"
    )

    # BM25 rankings mined earlier by rag_eval/bm25_rank.py (all 1000 questions)
    bm25_inpool, bm25_corpus = {}, {}
    for line in open(RAG_EVAL / "bm25_inpool.jsonl"):
        d = json.loads(line)
        bm25_inpool[d["id"]] = d["ranking"]
    for line in open(RAG_EVAL / "bm25_corpus.jsonl"):
        d = json.loads(line)
        bm25_corpus[d["id"]] = d["top50"]
    dense = dense_top50({q["id"] for q in train_qs})

    key2doc = {
        (c["title"], re.sub(r"\s+", " ", c["text"]).strip()): c["docid"]
        for c in corpus.docs
    }

    def dedup(blocks):
        seen, out = set(), []
        for b in blocks:
            if b and b not in seen:
                seen.add(b)
                out.append(b)
        return out

    items = []
    stats = Counter()
    for q in train_qs:
        qid, qtext = q["id"], q["question"]
        lang = lang_of(qid)
        qhead = f"问题:{qtext}" if lang == "zh" else f"Question: {qtext}"
        gold = set(q["gold_idx"])
        paras = {idx: (t, x) for idx, t, x in q["paragraphs"]}
        pool_docids = set()
        for idx, (t, x) in paras.items():
            pool_docids.add(key2doc.get((t, re.sub(r"\s+", " ", x).strip())))

        # ---- negative pools
        inpool_rank = bm25_inpool[qid]  # best-first pidx list
        hard_inpool = [i for i in inpool_rank if i not in gold]
        rank_pos = {pidx: r for r, pidx in enumerate(inpool_rank)}
        corpus_hard = [d for d in bm25_corpus[qid] if d not in pool_docids]
        dense_hard = [d for d in dense.get(qid, []) if d not in pool_docids]
        if not dense_hard:  # fallback keeps hard volume when service is down
            dense_hard = [d for d in corpus_hard if d not in corpus_hard[:3]][:3]
        easy = random_corpus_negatives(
            type("DS", (), {"corpus": corpus.title_text})(),
            qtext,
            {paras[g][0] for g in gold},
            q["answer"],
            2,
            rng,
        )

        # ---- A. pointwise relevance over all in-pool paragraphs
        for idx, (t, x) in sorted(paras.items()):
            is_gold = idx in gold
            hard = (not is_gold) and rank_pos[idx] < 10
            items.append(
                make_item(
                    qid,
                    "passage_relevance",
                    f"p{idx}",
                    "noul",
                    f"{qhead}\n\n{passage_block(t, x)}",
                    None,
                    PROP_RELEVANCE[lang],
                    ["yes", "no"],
                    {"yes": float(is_gold), "no": float(not is_gold)},
                    {
                        "source": "inpool",
                        "bm25_rank": rank_pos[idx],
                        "hard": bool(hard),
                        "title": t,
                    },
                )
            )
            stats[f"pt_{'yes' if is_gold else ('no_hard' if hard else 'no_easy')}"] += 1
        # ---- B. pointwise corpus-level hard + random easy negatives
        for src, docs in (("corpus_bm25", corpus_hard[:3]), ("dense", dense_hard[:3])):
            for rank, docid in enumerate(docs):
                c = corpus.by_docid[docid]
                items.append(
                    make_item(
                        qid,
                        "passage_relevance",
                        f"{src}{rank}",
                        "noul",
                        f"{qhead}\n\n{passage_block(c['title'], c['text'])}",
                        None,
                        PROP_RELEVANCE[lang],
                        ["yes", "no"],
                        {"yes": 0.0, "no": 1.0},
                        {
                            "source": src,
                            "hard": True,
                            "title": c["title"],
                            "docid": docid,
                        },
                    )
                )
                stats["pt_no_hard"] += 1
        for rank, (t, x) in enumerate(easy):
            items.append(
                make_item(
                    qid,
                    "passage_relevance",
                    f"rand{rank}",
                    "noul",
                    f"{qhead}\n\n{passage_block(t, x)}",
                    None,
                    PROP_RELEVANCE[lang],
                    ["yes", "no"],
                    {"yes": 0.0, "no": 1.0},
                    {"source": "random_filtered", "hard": False, "title": t},
                )
            )
            stats["pt_no_easy"] += 1

        # ---- C. listwise choice sets (7 variants)
        hard_blocks = [passage_block(paras[i][0], paras[i][1]) for i in hard_inpool[:8]]
        corpus_blocks = [
            passage_block(corpus.by_docid[d]["title"], corpus.by_docid[d]["text"])
            for d in corpus_hard[:3]
        ]
        dense_blocks = [
            passage_block(corpus.by_docid[d]["title"], corpus.by_docid[d]["text"])
            for d in dense_hard[:3]
        ]
        easy_blocks = [passage_block(t, x) for t, x in easy]
        gold_list = sorted(gold)
        spec = [
            (0, 1, hard_blocks[:4]),
            (1, 1, hard_blocks[4:8]),
            (2, 1, corpus_blocks[:2] + dense_blocks[:2]),
            (3, 1, dense_blocks[:1] + corpus_blocks[2:3] + hard_blocks[:2]),
            (4, 2, hard_blocks[1:3] + dense_blocks[1:3]),
            (
                5,
                1,
                dense_blocks[:4]
                if len(dense_blocks) >= 4
                else dense_blocks + hard_blocks[: 4 - len(dense_blocks)],
            ),
            (6, 2, corpus_blocks[:2] + hard_blocks[4:5] + easy_blocks[:1]),
        ]
        for variant, n_gold, negs in spec:
            n_gold = min(n_gold, len(gold_list))
            chosen_gold = rng.sample(gold_list, n_gold)
            for _ in range(5):  # avoid identical gold blocks (same title+prefix)
                blocks_try = [
                    passage_block(paras[g][0], paras[g][1]) for g in chosen_gold
                ]
                if len(set(blocks_try)) == len(blocks_try):
                    break
                chosen_gold = rng.sample(gold_list, n_gold)
            gold_blocks = [passage_block(paras[g][0], paras[g][1]) for g in chosen_gold]
            if len(set(gold_blocks)) < len(gold_blocks):  # give up on multi-gold
                chosen_gold, gold_blocks = chosen_gold[:1], gold_blocks[:1]
                n_gold = 1
            negs = [b for b in dedup(negs) if b not in set(gold_blocks)]
            if len(negs) < 4:  # pad from remaining hard in-pool blocks
                pad = [
                    b
                    for b in hard_blocks
                    if b not in negs and b not in set(gold_blocks)
                ]
                negs = dedup(negs + pad)
            negs = negs[:4]  # 1 gold + 4 neg -> K=5; 2 gold + 4 neg -> K=6
            options = gold_blocks + negs
            rng.shuffle(options)
            n_easy = sum(1 for b in negs if b in set(easy_blocks))
            t_mass = 1.0 / n_gold
            target = {
                c: (
                    t_mass
                    if c
                    in {passage_block(paras[g][0], paras[g][1]) for g in chosen_gold}
                    else 0.0
                )
                for c in options
            }
            items.append(
                make_item(
                    qid,
                    "evidence_choice",
                    f"v{variant}",
                    "choice",
                    qhead,
                    INSTR_CHOICE[lang],
                    None,
                    options,
                    target,
                    {
                        "k": len(options),
                        "n_gold": n_gold,
                        "n_neg": len(negs),
                        "n_easy": n_easy,
                        "gold_titles": [paras[g][0] for g in chosen_gold],
                    },
                )
            )
            stats[f"ch_K{len(options)}"] += 1

    # ---- global hard-negative ratio assertion (>= 50% of all negatives)
    neg_hard = neg_easy = 0
    for i in items:
        if i["mechanism"] == "rag_passage_relevance":
            if i["target_distribution"]["no"] == 1.0:
                if i["meta"]["hard"]:
                    neg_hard += 1
                else:
                    neg_easy += 1
        else:
            neg_hard += i["meta"]["n_neg"] - i["meta"]["n_easy"]
            neg_easy += i["meta"]["n_easy"]
    frac = neg_hard / (neg_hard + neg_easy)
    assert frac >= 0.5, f"hard negative fraction {frac:.3f} < 0.5"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    splits = {q["id"]: rag_split_of(q["id"]) for q in train_qs}
    with open(SPLITS_OUT, "w") as f:
        json.dump(splits, f, indent=1)

    print(f"wrote {OUT}: {len(items)} items")
    print("mechanisms:", dict(Counter(i["mechanism"] for i in items)))
    print("splits:", dict(Counter(i["split"] for i in items)))
    print(f"negatives: hard={neg_hard} easy={neg_easy} hard_frac={frac:.3f}")
    print("pointwise labels:", {k: v for k, v in stats.items() if k.startswith("pt_")})
    print("choice sets:", {k: v for k, v in stats.items() if k.startswith("ch_")})
    k_dist = Counter(len(i["candidates"]) for i in items)
    print("candidates per item:", dict(sorted(k_dist.items())))


if __name__ == "__main__":
    main()
