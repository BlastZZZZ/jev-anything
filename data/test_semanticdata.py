"""Tests for data/semantic_v1.jsonl: truth mapping, split leakage, length
distribution, class balance, and negative-filter logic.

Requires the generated data/semantic_v1.jsonl (run `python data/make_semanticdata.py`)
and the source RAG datasets ($XIAOJEV_RAG_DATA); skips cleanly without them.
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_semanticdata import Dataset, split_of, tokens, SNIPPET  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "semantic_v1.jsonl"
RAG_ROOT = os.environ.get("XIAOJEV_RAG_DATA")


@pytest.fixture(scope="module")
def items():
    if not DATA.exists():
        pytest.skip(f"{DATA} not generated (run data/make_semanticdata.py)")
    if not RAG_ROOT:
        pytest.skip("XIAOJEV_RAG_DATA not set (source QA datasets needed for gold checks)")
    rows = [json.loads(l) for l in open(DATA)]
    assert len({i["id"] for i in rows}) == len(rows), "duplicate ids"
    return rows


@pytest.fixture(scope="module")
def datasets(items):
    return {n: Dataset(n, RAG_ROOT) for n in ("hotpotqa", "2wikimultihopqa", "musique")}


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def in_state(cand_or_state, ptext):
    return norm(ptext)[:120] in norm(cand_or_state)


# 1. truth spot check on a deterministic sample
def test_truth_spot_check(items, datasets):
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    rng = random.Random(0)
    for item in rng.sample(items, 400):
        ds = datasets[item["meta"]["dataset"]]
        qid = item["meta"]["question_id"]
        gold_ids = ds.gold_pids(qid)
        gold_passages = [(t, x) for pid, t, x in ds.passages(qid) if pid in set(gold_ids or [])]
        mech = item["mechanism"]
        t = item["target_distribution"]
        check(abs(sum(t.values()) - 1) < 1e-9, f"{item['id']}: target not normalized")
        if mech == "sem_evidence_choice":
            winner = item["candidates"][[t[c] for c in item["candidates"]].index(1.0)]
            check(any(in_state(winner, x) for _, x in gold_passages),
                  f"{item['id']}: choice winner not a gold passage")
            n_gold = sum(1 for c in item["candidates"] if any(in_state(c, x) for _, x in gold_passages))
            check(n_gold == 1, f"{item['id']}: {n_gold} gold passages among options")
        elif mech == "sem_passage_relevance":
            state = item["state"]
            is_gold = any(in_state(state, x) for _, x in gold_passages)
            check(is_gold == (t["yes"] == 1.0), f"{item['id']}: relevance label vs gold mismatch")
        elif mech in ("sem_answerability", "sem_sufficiency"):
            state = item["state"]
            if item["meta"]["construction"] == "removed gold passage":
                check(t["no"] == 1.0, f"{item['id']}: removal variant must be no")
                removed = item["meta"]["removed_title"]
                check(removed in {g[0] for g in gold_passages},
                      f"{item['id']}: removed title not gold")
                n_still = sum(1 for g, x in gold_passages if g == removed and in_state(state, x))
                total_removed_title = sum(1 for g, x in gold_passages if g == removed)
                check(n_still < total_removed_title,
                      f"{item['id']}: removed passage text still in state")
            elif item["meta"]["construction"] == "full context":
                check(t["yes"] == 1.0, f"{item['id']}: full-context variant must be yes")
                for g, x in gold_passages:
                    check(in_state(state, x), f"{item['id']}: gold passage missing from full context")
    assert not failures, "\n".join(failures[:20])


# 2. split leakage: no question id in two splits
def test_split_leakage(items):
    q_splits = defaultdict(set)
    for i in items:
        q_splits[(i["meta"]["dataset"], i["meta"]["question_id"])].add(i["split"])
    bad = [q for q, s in q_splits.items() if len(s) > 1]
    assert not bad, f"question leakage across splits: {bad[:5]}"
    for (ds_name, qid), s in q_splits.items():
        assert s == {split_of(ds_name, qid)}, f"{ds_name}:{qid} split mismatch vs hash rule"


# 3. length distribution (chars; tokens checked separately in smoke)
def test_length_distribution(items):
    def stat(vals):
        vals = sorted(vals)
        return {"min": vals[0], "p50": vals[len(vals) // 2], "p95": vals[int(len(vals) * .95)], "max": vals[-1]}

    by_mech_len = defaultdict(list)
    for i in items:
        total = len(i["state"]) + sum(len(c) for c in i["candidates"])
        by_mech_len[i["mechanism"]].append(total)
    print("length chars by mechanism (state+candidates):")
    for m, v in sorted(by_mech_len.items()):
        print(f"  {m:24s} {stat(v)}")
    assert max(len(i["state"]) for i in items if i["mechanism"] == "sem_evidence_choice") < 2000, \
        "evidence_choice state too long"


# 4. class balance
def test_class_balance(items):
    mech_counts = Counter(i["mechanism"] for i in items)
    print("mechanism counts:", dict(mech_counts))
    ds_counts = Counter(i["meta"]["dataset"] for i in items)
    print("dataset counts:", dict(ds_counts))
    assert all(v >= 4000 for v in ds_counts.values()), "a dataset has <4000 items"
    for mech in ("sem_passage_relevance", "sem_answerability", "sem_sufficiency"):
        yes = sum(1 for i in items if i["mechanism"] == mech and i["target_distribution"]["yes"] == 1.0)
        no = sum(1 for i in items if i["mechanism"] == mech and i["target_distribution"]["no"] == 1.0)
        ratio = yes / max(no, 1)
        print(f"  {mech}: yes={yes} no={no} ratio={ratio:.3f}")
        assert 0.8 <= ratio <= 1.25, f"{mech}: yes/no imbalance {ratio:.2f}"


# 5. negative filter logic on ALL relevance negatives
def test_negative_filter(items, datasets):
    for i in items:
        if i["mechanism"] != "sem_passage_relevance" or i["target_distribution"]["no"] != 1.0:
            continue
        ds = datasets[i["meta"]["dataset"]]
        qid = i["meta"]["question_id"]
        q = ds.question(qid)
        neg_title = i["meta"]["neg_title"]
        assert not tokens(neg_title) & tokens(q), f"{i['id']}: negative title overlaps question"
        assert neg_title not in {t for t, _ in (ds.gold[qid].get("supporting_facts") or [])}, \
            f"{i['id']}: negative is a gold title"
    # every question still has its positive
    pos_q = {i["meta"]["question_id"] for i in items
             if i["mechanism"] == "sem_passage_relevance" and i["target_distribution"]["yes"] == 1.0}
    neg_q = {i["meta"]["question_id"] for i in items
             if i["mechanism"] == "sem_passage_relevance" and i["target_distribution"]["no"] == 1.0}
    print(f"relevance: questions with pos={len(pos_q)}, with neg={len(neg_q)} (filtered out {len(pos_q - neg_q)})")
