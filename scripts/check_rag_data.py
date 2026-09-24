"""Tests for data/rag_v1.jsonl: schema vs train.py loader, label correctness,
hard-negative ratio, split rules, and eval-set leakage. Run:
  python3 test_ragdata.py
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "data")]
from make_ragdata import rag_split_of, lang_of, DENSE_CACHE  # noqa
from make_semanticdata import split_of as sem_split_of  # noqa
from rag_eval.common import load_questions  # noqa

DATA = ROOT / "data/rag_v1.jsonl"
SPLITS = ROOT / "data/rag_v1.splits.json"

items = [json.loads(l) for l in open(DATA)]
by_id = {i["id"]: i for i in items}
assert len(by_id) == len(items), "duplicate ids"
questions = {q["id"]: q for q in load_questions()}
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def in_text(hay, ptext):
    return norm(ptext)[:120] in norm(hay)


REQUIRED_KEYS = {
    "id",
    "family",
    "mechanism",
    "primitive",
    "state",
    "instruction",
    "proposition",
    "candidates",
    "target_distribution",
    "difficulty",
    "split",
    "paired_id",
    "meta",
}

# 1. schema + target normalization
for i in items:
    check(REQUIRED_KEYS <= set(i), f"{i['id']}: missing keys")
    t = i["target_distribution"]
    check(set(t) == set(i["candidates"]), f"{i['id']}: target keys != candidates")
    check(abs(sum(t.values()) - 1) < 1e-9, f"{i['id']}: target not normalized")
    check(all(0 <= v <= 1 for v in t.values()), f"{i['id']}: target out of range")
    if i["primitive"] == "noul":
        check(i["candidates"] == ["yes", "no"], f"{i['id']}: noul candidates")
        check(i["proposition"], f"{i['id']}: noul needs proposition")
    else:
        check(5 <= len(i["candidates"]) <= 6, f"{i['id']}: choice K out of range")
        check(i["instruction"], f"{i['id']}: choice needs instruction")
        check(
            len(set(i["candidates"])) == len(i["candidates"]),
            f"{i['id']}: duplicate candidate strings",
        )

# 2. eval-set leakage: every question must be in the musique TRAIN split
eval_qids = {q for q in questions if sem_split_of("musique", q) != "train"}
check(
    len(eval_qids) == 293, f"expected 293 held-out eval questions, got {len(eval_qids)}"
)
used = {i["meta"]["question_id"] for i in items}
leak = used & eval_qids
check(not leak, f"EVAL LEAKAGE: {len(leak)} eval questions present: {sorted(leak)[:5]}")
check(len(used) == 707, f"expected 707 train questions, got {len(used)}")
for q in used:
    check(sem_split_of("musique", q) == "train", f"{q}: not musique train split")

# 3. rag split rule consistency + manifest
splits_manifest = json.load(open(SPLITS))
check(set(splits_manifest) == used, "splits manifest question set mismatch")
q_splits = defaultdict(set)
for i in items:
    q_splits[i["meta"]["question_id"]].add(i["split"])
bad = [q for q, s in q_splits.items() if len(s) > 1]
check(not bad, f"question across multiple rag splits: {bad[:5]}")
for q, s in q_splits.items():
    check(s == {rag_split_of(q)} == {splits_manifest[q]}, f"{q}: rag split mismatch")

# 4. label correctness on a deterministic sample (text-verified against raw data)
rng = random.Random(0)
gold_text = {}
for q in used:
    gold_text[q] = [
        x
        for idx, t, x in questions[q]["paragraphs"]
        if idx in set(questions[q]["gold_idx"])
    ]
for item in rng.sample(items, 500):
    qid = item["meta"]["question_id"]
    gtexts = gold_text[qid]
    if item["mechanism"] == "rag_passage_relevance":
        src = item["meta"]["source"]
        is_gold = any(in_text(item["state"], x) for x in gtexts)
        if item["target_distribution"]["yes"] == 1.0:
            check(
                is_gold and src == "inpool",
                f"{item['id']}: positive not an in-pool gold passage",
            )
        else:
            check(
                not is_gold or src == "inpool",
                f"{item['id']}: corpus/random negative contains a gold passage",
            )
            if src != "inpool":
                check(not is_gold, f"{item['id']}: corpus/random negative IS gold")
    else:
        winners = [c for c in item["candidates"] if item["target_distribution"][c] > 0]
        check(
            len(winners) == item["meta"]["n_gold"],
            f"{item['id']}: winner count != n_gold",
        )
        for w in winners:
            check(any(in_text(w, x) for x in gtexts), f"{item['id']}: winner not gold")
        losers = [c for c in item["candidates"] if item["target_distribution"][c] == 0]
        for l in losers:
            check(
                not any(in_text(l, x) for x in gtexts), f"{item['id']}: loser is gold"
            )
        check(
            abs(sum(item["target_distribution"].values()) - 1) < 1e-9,
            f"{item['id']}: choice target not normalized",
        )

# 5. hard-negative ratio (global and per source)
hard = easy = 0
per_src = Counter()
for i in items:
    if i["mechanism"] == "rag_passage_relevance":
        if i["target_distribution"]["no"] == 1.0:
            key = ("hard" if i["meta"]["hard"] else "easy", i["meta"]["source"])
            per_src[key] += 1
            if i["meta"]["hard"]:
                hard += 1
            else:
                easy += 1
    else:
        hard += i["meta"]["n_neg"] - i["meta"]["n_easy"]
        easy += i["meta"]["n_easy"]
frac = hard / (hard + easy)
check(frac >= 0.5, f"hard negative fraction {frac:.3f} < 0.5")

# 6. distribution stats
print("items:", len(items))
print("mechanisms:", dict(Counter(i["mechanism"] for i in items)))
print("splits:", dict(Counter(i["split"] for i in items)))
print("langs:", dict(Counter(i["meta"]["lang"] for i in items)))
print("pointwise negatives by (hardness, source):", dict(sorted(per_src.items())))
print(f"hard negatives: {hard}/{hard + easy} = {frac:.3f}")
print(
    "candidates per item:",
    dict(sorted(Counter(len(i["candidates"]) for i in items).items())),
)
noul_yes = sum(
    1
    for i in items
    if i["primitive"] == "noul" and i["target_distribution"]["yes"] == 1.0
)
noul_no = sum(
    1
    for i in items
    if i["primitive"] == "noul" and i["target_distribution"]["no"] == 1.0
)
print(
    f"pointwise yes:no = {noul_yes}:{noul_no} (1:{noul_no / noul_yes:.1f}; intentional counter-bias "
    "to the yes-saturation observed in rag_eval)"
)
choice_per_q = Counter(
    i["meta"]["question_id"] for i in items if i["mechanism"] == "rag_evidence_choice"
)
print("choice sets per question:", dict(Counter(choice_per_q.values())))
print("dense cache used:", DENSE_CACHE.exists())

if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures[:20]:
        print(" -", f)
    raise SystemExit(1)
print(
    f"ALL CHECKS PASSED ({len(items)} items, {len(used)} train questions, 0 eval leakage)"
)
