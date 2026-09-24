"""Shared data/prompt utilities for the xiaojev-v3 MuSiQue RAG evaluation.

Prompt formats mirror make_semanticdata.py exactly (English variants), since
v3 was trained on them. Contamination control: v3's semantic training data
included musique questions whose split_of() == 'train'; those 707 questions
are excluded from all v3 metrics here (dev+calibration+test = 293 remain).
"""

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("XIAOJEV_RAG_DATA", "datasets")) / "musique"
SPLIT_SEED = 20260922  # same as make_semanticdata.py
SNIPPET = 300

INSTR_CHOICE = "Which of the following passages contains the information needed to answer the question above?"
PROP_RELEVANCE = "The passage above helps answer the question."
PROP_ANSWERABLE = "Given only the provided passages, the question can be answered."


def split_of(qid):
    h = (
        int(hashlib.sha256(f"musique:{qid}:{SPLIT_SEED}".encode()).hexdigest()[:8], 16)
        % 100
    )
    return (
        "train" if h < 70 else "dev" if h < 80 else "calibration" if h < 90 else "test"
    )


def snippet(text):
    return re.sub(r"\s+", " ", text).strip()[:SNIPPET]


def passage_block(title, text):
    return f"[{title}] {snippet(text)}"


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_questions():
    raw = {r["id"]: r for r in json.load(open(ROOT / "raw" / "musique.json"))}
    gold = {g["id"]: g for g in map(json.loads, open(ROOT / "gold.jsonl"))}
    out = []
    for qid, r in raw.items():
        paras = [
            (int(p["idx"]), p["title"], p["paragraph_text"]) for p in r["paragraphs"]
        ]
        paras.sort(key=lambda x: x[0])
        gold_idx = sorted(
            int(p["idx"])
            for p in r["paragraphs"]
            if p["is_supporting"] in (True, "True")
        )
        hop = int(qid.split("hop")[0])
        out.append(
            {
                "id": qid,
                "question": r["question"],
                "paragraphs": paras,
                "gold_idx": gold_idx,
                "answer": gold[qid].get("answer") or "",
                "answer_aliases": gold[qid].get("answer_aliases") or [],
                "hop": hop,
                "split": split_of(qid),
            }
        )
    return out


def load_corpus():
    return [json.loads(l) for l in open(ROOT / "corpus.jsonl")]
