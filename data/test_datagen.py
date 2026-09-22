"""Self-consistency tests for datagen.py (VCDM P0)."""

import itertools
import random
from collections import Counter
from fractions import Fraction
from math import comb

import pytest

import datagen


@pytest.fixture(scope="module")
def data():
    return datagen.build(n_a=6000, n_b=3000, seed=123)


def sample(records, mech=None, n=150, pred=None):
    pool = [r for r in records if (mech is None or r["mechanism"] == mech)]
    if pred:
        pool = [r for r in pool if pred(r)]
    return random.Random(0).sample(pool, min(n, len(pool)))


# ---------------------------------------------------------- brute-force oracles

def urn_balls(params):
    balls = []
    for ci, n in zip(params["colors"], params["counts"]):
        balls += [ci] * n
    return balls


def urn_outcomes(params):
    balls = urn_balls(params)
    if params["draws"] == 1:
        return [(b,) for b in balls]
    if params["replace"]:
        return list(itertools.product(balls, repeat=2))
    return [(balls[i], balls[j]) for i in range(len(balls))
            for j in range(len(balls)) if i != j]


def eval_urn(params):
    ev = params["event"]
    if ev[0] == "dist":
        balls = urn_balls(params)
        t = len(balls)
        return [Fraction(params["counts"][i], t) for i in range(len(params["counts"]))]
    outs = urn_outcomes(params)
    def ok(o):
        if ev[0] == "single":
            return o[0] == ev[1]
        if ev[0] == "both":
            return all(x == ev[1] for x in o)
        if ev[0] == "at_least":
            return ev[1] in o
        if ev[0] == "same":
            return o[0] == o[1]
        if ev[0] == "ordered":
            return o[0] == ev[1] and o[1] == ev[2]
        raise AssertionError(ev)
    return Fraction(sum(1 for o in outs if ok(o)), len(outs))


def eval_lottery(params):
    K, ev = params["K"], params["event"]
    if ev[0] == "dist":
        return [Fraction(1, K)] * K
    def ok(x):
        if ev[0] == "eq":
            return x == ev[1]
        if ev[0] == "gt":
            return x > ev[1]
        if ev[0] == "parity":
            return (x % 2 == 0) == (ev[1] == "even")
        if ev[0] == "range":
            return ev[1] <= x <= ev[2]
        raise AssertionError(ev)
    return Fraction(sum(1 for x in range(1, K + 1) if ok(x)), K)


def eval_die(params):
    F = params["faces"]
    probs = datagen.die_probs(F, tuple(params["loaded"]) if params["loaded"] else None)
    ev = params["event"]
    if ev[0] == "dist":
        return probs
    def ok(x):
        if ev[0] == "face":
            return x == ev[1]
        if ev[0] == "parity":
            return (x % 2 == 1) == (ev[1] == "odd")
        if ev[0] == "prime":
            return x in datagen.PRIMES
        if ev[0] == "gt":
            return x > ev[1]
        raise AssertionError(ev)
    return sum(probs[x - 1] for x in range(1, F + 1) if ok(x))


def eval_coin(params):
    pct, n, ev = params["pct"], params["n"], params["event"]
    b = Fraction(pct, 100)
    if ev[0] == "dist":
        return [Fraction(comb(n, k) * pct ** k * (100 - pct) ** (n - k), 100 ** n)
                for k in range(n + 1)]
    if ev[0] == "heads":
        return b
    if ev[0] == "all":
        return b ** n
    if ev[0] == "at_least":
        return 1 - (1 - b) ** n
    if ev[0] == "exactly":
        k = ev[1]
        return Fraction(comb(n, k) * pct ** k * (100 - pct) ** (n - k), 100 ** n)
    raise AssertionError(ev)


def eval_spinner(params):
    ev = params["event"]
    if ev[0] == "dist":
        return [Fraction(a, 360) for a in params["angles"]]
    return Fraction(params["angles"][ev[1]], 360)


def eval_card(params):
    deck, ev = params["deck"], params["event"]
    total = sum(len(v) for v in deck.values())
    if ev[0] == "dist":
        return [Fraction(len(v), total) for v in deck.values()]
    if ev[0] == "color":
        red = ev[1] == "red"
        n = sum(len(v) for s, v in deck.items() if (s in datagen.RED_SUITS) == red)
        return Fraction(n, total)
    if ev[0] == "suit":
        return Fraction(len(deck[ev[1]]), total)
    if ev[0] == "rank":
        return Fraction(sum(1 for v in deck.values() if ev[1] in v), total)
    raise AssertionError(ev)


def eval_dice_sum(params):
    d, F, ev = params["d"], params["faces"], params["event"]
    faces = F if isinstance(F, list) else [F] * d
    outs = itertools.product(*[range(1, f + 1) for f in faces])
    sums = [sum(o) for o in outs]
    total = 1
    for f in faces:
        total *= f
    if ev[0] == "dist":
        lo = d
        return [Fraction(sum(1 for s in sums if s == lo + i), total)
                for i in range(sum(faces) - d + 1)]
    if ev[0] == "eq":
        return Fraction(sum(1 for s in sums if s == ev[1]), total)
    if ev[0] == "ge":
        return Fraction(sum(1 for s in sums if s >= ev[1]), total)
    raise AssertionError(ev)


EVAL = {"urn": eval_urn, "lottery": eval_lottery, "die": eval_die,
        "coin": eval_coin, "spinner": eval_spinner, "card": eval_card,
        "dice_sum": eval_dice_sum}


def check_oracle(r):
    p = r["meta"]["params"]
    got = EVAL[p["kind"]](p)
    if isinstance(got, list):
        assert len(got) == len(r["candidates"])
        for c, f in zip(r["candidates"], got):
            assert abs(r["target_distribution"][c] - float(f)) < 1e-9, r["id"]
    else:
        key = r["candidates"][0] if r["primitive"] == "choice" else "yes"
        assert abs(r["target_distribution"][key] - float(got)) < 1e-9, r["id"]


# ---------------------------------------------------------- tests

def test_schema(data):
    keys = {"id", "family", "mechanism", "primitive", "state", "instruction",
            "proposition", "candidates", "target_distribution", "difficulty",
            "split", "paired_id", "meta"}
    for r in data:
        assert keys <= set(r)
        assert r["family"] in {"random_mechanism", "deterministic"}
        assert r["primitive"] in {"choice", "noul"}
        assert r["split"] in datagen.SPLIT_ORDER
        if r["primitive"] == "choice":
            assert r["instruction"] and r["proposition"] is None
        else:
            assert r["proposition"] and r["instruction"] is None


def test_distribution_sums_to_one(data):
    for r in data:
        s = sum(r["target_distribution"].values())
        assert abs(s - 1.0) < 1e-6, (r["id"], s)


def test_candidates_match_distribution(data):
    for r in data:
        assert set(r["candidates"]) == set(r["target_distribution"]), r["id"]
        if r["primitive"] == "noul":
            assert r["candidates"] == ["yes", "no"]


def test_paired_consistency(data):
    by_id = {r["id"]: r for r in data}
    paired = [r for r in data if r["paired_id"]]
    assert paired, "no paired records generated"
    for r in paired:
        q = by_id[r["paired_id"]]
        assert q["paired_id"] == r["id"]
        assert q["split"] == r["split"]
        assert q["state"] == r["state"]
        assert {r["primitive"], q["primitive"]} == {"choice", "noul"}
        choice = r if r["primitive"] == "choice" else q
        noul = q if r["primitive"] == "choice" else r
        assert abs(choice["target_distribution"][choice["candidates"][0]]
                   - noul["target_distribution"]["yes"]) < 1e-9


def test_no_cross_split_duplicates(data):
    seen = {}
    for r in data:
        h = datagen.text_hash(r)
        if h in seen:
            assert seen[h] == r["split"], (r["id"], seen[h], r["split"])
        seen[h] = r["split"]


def test_ood_isolation(data):
    for r in data:
        if r["split"] == "ood":
            assert r["mechanism"] in datagen.OOD_MECHS, r["id"]
        else:
            assert r["mechanism"] not in datagen.OOD_MECHS, r["id"]
    assert any(r["split"] == "ood" for r in data)


def test_bruteforce_enumerative(data):
    for r in sample(data, "dice_sum"):
        check_oracle(r)
    for r in sample(data, "urn", pred=lambda r: r["difficulty"] == "enumerative"):
        check_oracle(r)


def test_bruteforce_simple(data):
    for mech in ["urn", "lottery", "die", "coin", "spinner", "card"]:
        recs = sample(data, mech, n=100,
                      pred=lambda r: r["difficulty"] == "simple")
        assert recs, mech
        for r in recs:
            check_oracle(r)


def test_deterministic_one_hot(data):
    det = [r for r in data if r["family"] == "deterministic"]
    assert det
    for r in det:
        vals = sorted(r["target_distribution"].values(), reverse=True)
        assert vals[0] == 1.0 and all(v == 0.0 for v in vals[1:]), r["id"]


def test_ratios(data):
    a = [r for r in data if r["family"] == "random_mechanism"]
    b = [r for r in data if r["family"] == "deterministic"]
    assert len(a) == pytest.approx(6000, rel=0.05)
    assert len(b) == pytest.approx(3000, rel=0.05)
    non_ood = [r for r in data if r["split"] != "ood"]
    cnt = Counter(r["split"] for r in non_ood)
    total = len(non_ood)
    assert 0.60 <= cnt["train"] / total <= 0.80
    for s in ("dev", "calibration", "test"):
        assert 0.05 <= cnt[s] / total <= 0.18, s
    paired = sum(1 for r in a if r["paired_id"])
    assert 0.75 <= paired / len(a) <= 0.90
    zh = sum(1 for r in data if r["meta"]["lang"] == "zh")
    assert 0.6 <= zh / len(data) <= 0.8


def test_reproducible(data):
    again = datagen.build(n_a=6000, n_b=3000, seed=123)
    assert [r["id"] for r in again] == [r["id"] for r in data]
    assert [datagen.text_hash(r) for r in again] == [datagen.text_hash(r) for r in data]
