"""VCDM P0: programmatic training-data generator.

Family A (random_mechanism): the generator knows the mechanism parameters, so
target_distribution is the exact analytic/enumerated value (Fractions
internally). Every binary event emits a paired Choice + Noul couplet
(paired_id cross-referenced). Family B (deterministic): one-hot targets.
card and dice_sum live entirely in the ood split. No LLM calls.
"""

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from fractions import Fraction

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19}
SPLIT_ORDER = ["train", "dev", "calibration", "test", "ood"]
OOD_MECHS = {"card", "dice_sum"}

COLORS = [("红", "red"), ("蓝", "blue"), ("黑", "black"),
          ("白", "white"), ("绿", "green"), ("黄", "yellow")]
SUITS = [("红桃", "hearts", True), ("黑桃", "spades", False),
         ("方块", "diamonds", True), ("梅花", "clubs", False)]
RED_SUITS = {"hearts", "diamonds"}
RANKS = ["A"] + [str(i) for i in range(2, 11)] + ["J", "Q", "K"]

BIN_INSTR = {
    "zh": ["下列哪种情况会发生？", "实际发生的会是哪个结果？", "哪个结果会发生？"],
    "en": ["Which of the following outcomes will occur?",
           "Which outcome occurs?", "What will the outcome be?"],
}

# Reference quotas at n_a=60000 (binary events emit 2 records each).
A_BIN_Q = {"urn": 4500, "lottery": 3600, "die": 4500, "coin": 4500,
           "spinner": 3600, "card": 2000, "dice_sum": 1800}
A_MUL_Q = {"urn": 1500, "lottery": 1000, "die": 2000, "coin": 1500,
           "spinner": 1500, "card": 1000, "dice_sum": 300}
B_Q = {"arithmetic": 6000, "comparison": 6000, "membership": 6000,
       "date": 6000, "string": 6000}


def pick_lang(rng):
    return "zh" if rng.random() < 0.7 else "en"


def cname(ci, lang):
    return COLORS[ci][0] if lang == "zh" else COLORS[ci][1]


def rec(mech, prim, state, question, cands, dist, diff, lang, params):
    return {
        "mechanism": mech,
        "primitive": prim,
        "state": state,
        "instruction": question if prim == "choice" else None,
        "proposition": question if prim == "noul" else None,
        "candidates": cands,
        "target": dist,
        "difficulty": diff,
        "meta": {"lang": lang, "params": params},
    }


def mk_pair(rng, mech, state, instr_opts, cands, prop, p, diff, lang, params):
    instr = rng.choice(instr_opts or BIN_INSTR[lang])
    p = Fraction(p)
    choice_r = rec(mech, "choice", state, instr, cands,
                   {cands[0]: p, cands[1]: 1 - p}, diff, lang, params)
    noul_r = rec(mech, "noul", state, prop, ["yes", "no"],
                 {"yes": p, "no": 1 - p}, diff, lang, params)
    return {"mech": mech, "pair": True, "recs": [choice_r, noul_r]}


def mk_single(mech, prim, state, question, cands, dist, diff, lang, params):
    return {"mech": mech, "pair": False,
            "recs": [rec(mech, prim, state, question, cands, dist, diff, lang, params)]}


def join_zh(parts):
    return "、".join(parts)


def join_en(parts):
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------- urn

def urn_state(rng, lang, cis, counts, total, draws, replace):
    items = list(zip(cis, counts))
    rng.shuffle(items)
    if lang == "zh":
        p1 = join_zh([f"{n} 个{cname(ci, lang)}色球" for ci, n in items])
        p2 = "，".join([f"{cname(ci, lang)}色球 {n} 个" for ci, n in items])
        p3 = "，".join([f"{cname(ci, lang)}色 {n} 个" for ci, n in items])
        if draws == 1:
            draw = "从中随机抽出一个球。"
        elif replace:
            draw = "先随机摸出一个球，记录颜色后放回，再随机摸出一个球。"
        else:
            draw = "从中随机连续摸出两个球，摸出的球不放回。"
        t = rng.choice([
            f"一个瓮中共有 {total} 个球：{p1}。",
            f"袋子里有 {p1}，一共 {total} 个球，除颜色外完全相同。",
            f"不透明的罐子里装着 {total} 个球，其中 {p3}。",
            f"一只瓮里放着 {p2}，共 {total} 个球，大小质地都相同。",
            f"桌上有个不透明容器，内有 {total} 个球（{p3}）。",
        ])
    else:
        p4 = join_en([f"{n} {cname(ci, lang)} balls" for ci, n in items])
        p5 = ", ".join([f"{n} {cname(ci, lang)}" for ci, n in items])
        if draws == 1:
            draw = "One ball is drawn at random."
        elif replace:
            draw = ("A ball is drawn at random, put back, "
                    "and then a ball is drawn at random again.")
        else:
            draw = ("Two balls are drawn at random, one after another, "
                    "without replacement.")
        t = rng.choice([
            f"An urn contains {p4}, {total} balls in total. ",
            f"There are {total} balls in an opaque urn: {p5}. ",
        ])
    return t + draw


def gen_urn_binary(rng):
    ncol = rng.randint(2, 5)
    cis = rng.sample(range(len(COLORS)), ncol)
    counts = [rng.randint(1, 20) for _ in cis]
    total = sum(counts)
    if total < 2 or total > 100 or total % 10 == 0:
        return None
    draws = rng.choice([1, 2, 2])
    replace = rng.random() < 0.5
    lang = pick_lang(rng)
    state = urn_state(rng, lang, cis, counts, total, draws, replace)
    cnt = dict(zip(cis, counts))
    params = {"kind": "urn", "colors": cis, "counts": counts,
              "draws": draws, "replace": replace}
    if draws == 1:
        ci = rng.choice(cis)
        c = cname(ci, lang)
        p = Fraction(cnt[ci], total)
        if lang == "zh":
            cands = [f"{c}色球", f"非{c}色球"]
            prop = rng.choice([f"抽出的球是{c}色的。", f"抽出的这个球是{c}色球。"])
        else:
            cands = [c, f"not {c}"]
            prop = rng.choice([f"The drawn ball is {c}.", f"The ball drawn is {c}."])
        params["event"] = ["single", ci]
        return mk_pair(rng, "urn", state, None, cands, prop, p, "simple", lang, params)
    kind = rng.choice(["both", "at_least", "same", "ordered"])
    diff = "enumerative"
    if kind == "same":
        if replace:
            p = sum(Fraction(n * n, total * total) for n in counts)
        else:
            if all(n < 2 for n in counts):
                return None
            p = sum(Fraction(n * (n - 1), total * (total - 1)) for n in counts)
        cands = ["颜色相同", "颜色不同"] if lang == "zh" else ["same color", "different colors"]
        prop = ("抽出的两个球颜色相同。" if lang == "zh"
                else "The two drawn balls have the same color.")
        params["event"] = ["same"]
    elif kind == "ordered":
        c1, c2 = rng.sample(cis, 2)
        n1, n2 = cnt[c1], cnt[c2]
        den = total * total if replace else total * (total - 1)
        p = Fraction(n1 * n2, den)
        a, b = cname(c1, lang), cname(c2, lang)
        if lang == "zh":
            cands = [f"先{a}后{b}", "其他结果"]
            prop = f"第一次摸出{a}色球，且第二次摸出{b}色球。"
        else:
            cands = [f"{a} then {b}", "anything else"]
            prop = f"The first ball is {a} and the second ball is {b}."
        params["event"] = ["ordered", c1, c2]
    else:
        ci = rng.choice(cis)
        n = cnt[ci]
        if not replace and kind == "both" and n < 2:
            return None
        if replace:
            p = Fraction(n * n, total * total) if kind == "both" else \
                1 - Fraction((total - n) ** 2, total * total)
        else:
            p = Fraction(n * (n - 1), total * (total - 1)) if kind == "both" else \
                1 - Fraction((total - n) * (total - n - 1), total * (total - 1))
        c = cname(ci, lang)
        if lang == "zh":
            cands = ([f"两个都是{c}色", f"不都是{c}色"] if kind == "both"
                     else [f"至少一个{c}色", f"没有{c}色"])
            prop = (f"摸出的两个球都是{c}色的。" if kind == "both"
                    else f"两次中至少有一次摸出{c}色球。")
        else:
            cands = ([f"both {c}", f"not both {c}"] if kind == "both"
                     else [f"at least one {c}", f"no {c}"])
            prop = (f"Both drawn balls are {c}." if kind == "both"
                    else f"At least one of the two drawn balls is {c}.")
        params["event"] = [kind, ci]
    return mk_pair(rng, "urn", state, None, cands, prop, p, diff, lang, params)


def gen_urn_multi(rng):
    ncol = rng.randint(2, 5)
    cis = rng.sample(range(len(COLORS)), ncol)
    counts = [rng.randint(1, 20) for _ in cis]
    total = sum(counts)
    if total < 2 or total > 100 or total % 10 == 0:
        return None
    lang = pick_lang(rng)
    state = urn_state(rng, lang, cis, counts, total, 1, False)
    if lang == "zh":
        cands = [f"{cname(ci, lang)}色" for ci in cis]
        instr = rng.choice(["抽出球的颜色会是什么？", "抽出的这个球是什么颜色？"])
    else:
        cands = [cname(ci, lang) for ci in cis]
        instr = rng.choice(["What color will the drawn ball be?",
                            "What is the color of the drawn ball?"])
    dist = {f"{cname(ci, lang)}色" if lang == "zh" else cname(ci, lang):
            Fraction(n, total) for ci, n in zip(cis, counts)}
    params = {"kind": "urn", "colors": cis, "counts": counts,
              "draws": 1, "replace": False, "event": ["dist"]}
    return mk_single("urn", "choice", state, instr, cands, dist, "simple", lang, params)


# ---------------------------------------------------------------- lottery

def lottery_state(rng, lang, K):
    if lang == "zh":
        return rng.choice([
            f"一个抽奖箱中有 {K} 个球，分别编号 1 到 {K}，每个球被抽中的概率完全相同。随机抽出一个球。",
            f"抽奖机里放着编号 1 至 {K} 的 {K} 个小球，搅匀后随机弹出一个，每个号码等概率。",
            f"一场公平抽签：共 {K} 张签，编号从 1 到 {K}，随机抽出一张。",
            f"箱子里有 {K} 个一模一样的球，编号依次为 1、2、……、{K}，闭眼随机摸出一个。",
        ])
    return rng.choice([
        f"A lottery machine holds {K} balls numbered 1 through {K}; one is drawn uniformly at random.",
        f"There are {K} tickets numbered 1 to {K} in a box, each equally likely; one ticket is drawn.",
    ])


def gen_lottery_binary(rng):
    K = rng.randint(2, 100)
    lang = pick_lang(rng)
    state = lottery_state(rng, lang, K)
    kind = rng.choice(["eq", "gt", "parity", "range"])
    params = {"kind": "lottery", "K": K}
    if kind == "eq":
        m = rng.randint(1, K)
        p = Fraction(1, K)
        if lang == "zh":
            cands = [f"{m} 号", f"非 {m} 号"]
            prop = rng.choice([f"抽中的号码是 {m}。", f"抽到的号码恰好是 {m} 号。"])
        else:
            cands = [f"number {m}", "any other number"]
            prop = rng.choice([f"The drawn number is {m}.", f"Number {m} is drawn."])
        params["event"] = ["eq", m]
    elif kind == "gt":
        x = rng.randint(1, K - 1)
        p = Fraction(K - x, K)
        if lang == "zh":
            cands = [f"大于 {x}", f"不超过 {x}"]
            prop = f"抽中的号码大于 {x}。"
        else:
            cands = [f"greater than {x}", f"at most {x}"]
            prop = f"The drawn number is greater than {x}."
        params["event"] = ["gt", x]
    elif kind == "parity":
        even = rng.random() < 0.5
        n_even = K // 2
        p = Fraction(n_even if even else K - n_even, K)
        if lang == "zh":
            cands = ["偶数", "奇数"] if even else ["奇数", "偶数"]
            prop = "抽中的号码是偶数。" if even else "抽中的号码是奇数。"
        else:
            cands = ["even", "odd"] if even else ["odd", "even"]
            prop = "The drawn number is even." if even else "The drawn number is odd."
        params["event"] = ["parity", "even" if even else "odd"]
    else:
        width = rng.randint(2, max(2, K // 2))
        a = rng.randint(1, K - width + 1)
        b = a + width - 1
        p = Fraction(width, K)
        if lang == "zh":
            cands = [f"{a}–{b} 之间", "其他号码"]
            prop = f"抽中的号码在 {a} 到 {b} 之间（含端点）。"
        else:
            cands = [f"between {a} and {b}", "outside that range"]
            prop = f"The drawn number is between {a} and {b} inclusive."
        params["event"] = ["range", a, b]
    return mk_pair(rng, "lottery", state, None, cands, prop, p, "simple", lang, params)


def gen_lottery_multi(rng):
    K = rng.randint(2, 10)
    lo = rng.randint(1, 30)
    hi = lo + K - 1
    lang = pick_lang(rng)
    if lang == "zh":
        state = rng.choice([
            f"一个抽奖箱中有 {K} 个球，分别编号 {lo} 到 {hi}，每个球被抽中的概率完全相同。随机抽出一个球。",
            f"一场公平抽签：共 {K} 张签，编号从 {lo} 到 {hi}，随机抽出一张。",
            f"箱子里有 {K} 个一模一样的球，编号依次为 {lo} 到 {hi}，闭眼随机摸出一个。",
        ])
        style = rng.random() < 0.5
        cands = [f"{i} 号" if style else str(i) for i in range(lo, hi + 1)]
        instr = rng.choice(["抽中的号码是哪个？", "会抽中哪个号码？", "抽出的球编号是多少？"])
    else:
        state = rng.choice([
            f"A lottery machine holds {K} balls numbered {lo} through {hi}; one is drawn uniformly at random.",
            f"There are {K} tickets numbered {lo} to {hi} in a box, each equally likely; one ticket is drawn.",
        ])
        cands = [str(i) if rng.random() < 0.5 else f"#{i}" for i in range(lo, hi + 1)]
        instr = rng.choice(["Which number is drawn?", "What number comes out?"])
    dist = {c: Fraction(1, K) for c in cands}
    params = {"kind": "lottery", "K": K, "lo": lo, "event": ["dist"]}
    return mk_single("lottery", "choice", state, instr, cands, dist, "simple", lang, params)


# ---------------------------------------------------------------- die

DIE_FACES = [4, 6, 8, 10, 11, 12, 20]


def die_probs(F, loaded):
    if loaded is None:
        return [Fraction(1, F)] * F
    face, pct = loaded
    p = Fraction(pct, 100)
    rest = (1 - p) / (F - 1)
    return [p if i + 1 == face else rest for i in range(F)]


def die_state(rng, lang, F, loaded):
    if loaded is None:
        if lang == "zh":
            return rng.choice([
                f"一颗均匀的 {F} 面骰子，面分别标有 1 到 {F}，掷一次，每面等概率朝上。",
                f"一个公平的 {F} 面骰（编号 1–{F}）被掷出一次。",
                f"投掷一颗标准的 {F} 面骰子一次，各面点数 1 至 {F}，完全均匀。",
            ])
        return rng.choice([
            f"A fair {F}-sided die with faces numbered 1 to {F} is rolled once.",
            f"Someone rolls a balanced {F}-sided die (faces 1 through {F}) a single time.",
        ])
    face, pct = loaded
    pstr = f"{pct / 100:.2f}"
    if lang == "zh":
        return rng.choice([
            f"一颗灌铅的 {F} 面骰子：掷出 {face} 点的概率是 {pstr}，其余 {F - 1} 个面彼此等概率。掷一次。",
            f"一个不均匀的 {F} 面骰（面编号 1 到 {F}）：{face} 点朝上的概率为 {pstr}，剩余各面平分其余概率。掷一次。",
        ])
    return rng.choice([
        f"A loaded {F}-sided die (faces 1–{F}) lands on {face} with probability {pstr}; "
        f"the other faces share the remaining probability equally. It is rolled once.",
        f"An unfair {F}-sided die: face {face} comes up with probability {pstr}, "
        f"and the remaining {F - 1} faces are equally likely. One roll.",
    ])


def gen_die_binary(rng):
    F = rng.choice(DIE_FACES)
    loaded = None
    if rng.random() < 0.5:
        loaded = (rng.randint(1, F), rng.randint(2, 12) * 5)
    probs = die_probs(F, loaded)
    lang = pick_lang(rng)
    state = die_state(rng, lang, F, loaded)
    kind = rng.choice(["face", "parity", "prime", "gt"])
    params = {"kind": "die", "faces": F,
              "loaded": list(loaded) if loaded else None}
    if kind == "face":
        v = rng.randint(1, F)
        p = probs[v - 1]
        if lang == "zh":
            cands = [f"{v} 点", f"非 {v} 点"]
            prop = rng.choice([f"掷出的点数是 {v}。", f"朝上的面是 {v} 点。"])
        else:
            cands = [str(v), f"not {v}"]
            prop = rng.choice([f"The roll shows {v}.", f"The die comes up {v}."])
        params["event"] = ["face", v]
    elif kind == "parity":
        odd = rng.random() < 0.5
        p = sum(probs[i] for i in range(F) if (i + 1) % 2 == (1 if odd else 0))
        if lang == "zh":
            cands = ["奇数", "偶数"] if odd else ["偶数", "奇数"]
            prop = "掷出的点数是奇数。" if odd else "掷出的点数是偶数。"
        else:
            cands = ["odd", "even"] if odd else ["even", "odd"]
            prop = "The roll is odd." if odd else "The roll is even."
        params["event"] = ["parity", "odd" if odd else "even"]
    elif kind == "prime":
        p = sum(probs[i] for i in range(F) if (i + 1) in PRIMES)
        if lang == "zh":
            cands = ["素数", "非素数"]
            prop = rng.choice(["掷出的点数是素数。", "朝上一面的点数是质数。"])
        else:
            cands = ["prime", "not prime"]
            prop = "The roll is a prime number."
        params["event"] = ["prime"]
    else:
        x = rng.randint(1, F - 1)
        p = sum(probs[i] for i in range(F) if (i + 1) > x)
        if lang == "zh":
            cands = [f"大于 {x}", f"不超过 {x}"]
            prop = f"掷出的点数大于 {x}。"
        else:
            cands = [f"greater than {x}", f"at most {x}"]
            prop = f"The roll is greater than {x}."
        params["event"] = ["gt", x]
    return mk_pair(rng, "die", state, None, cands, prop, p, "simple", lang, params)


def gen_die_multi(rng):
    F = rng.choice(DIE_FACES)
    loaded = None
    if rng.random() < 0.5:
        loaded = (rng.randint(1, F), rng.randint(2, 12) * 5)
    probs = die_probs(F, loaded)
    lang = pick_lang(rng)
    state = die_state(rng, lang, F, loaded)
    if lang == "zh":
        style = rng.choice([lambda i: f"{i} 点", str, lambda i: f"第 {i} 面"])
        cands = [style(i) for i in range(1, F + 1)]
        instr = rng.choice(["掷出的点数是多少？", "朝上的一面是几点？", "会掷出哪一面？"])
    else:
        style = rng.choice([str, lambda i: f"face {i}"])
        cands = [style(i) for i in range(1, F + 1)]
        instr = rng.choice(["What number comes up?", "Which face lands up?"])
    dist = {c: probs[i] for i, c in enumerate(cands)}
    params = {"kind": "die", "faces": F,
              "loaded": list(loaded) if loaded else None, "event": ["dist"]}
    return mk_single("die", "choice", state, instr, cands, dist, "simple", lang, params)


# ---------------------------------------------------------------- coin

def coin_state(rng, lang, pct, n):
    b = f"{pct / 100:.2f}"
    if lang == "zh":
        toss = "投掷一次。" if n == 1 else f"独立投掷 {n} 次。"
        return rng.choice([
            f"一枚不均匀的硬币，每次投掷正面朝上的概率是 {b}，反面朝上的概率是 {1 - pct / 100:.2f}。{toss}",
            f"有一枚偏重的硬币：掷出正面的概率为 {b}。现在{toss}",
            f"硬币正面概率 {b}、反面概率 {1 - pct / 100:.2f}，各次投掷相互独立。{toss}",
            f"一枚硬币被动过手脚，单次投掷正面概率固定为 {b}。{toss}",
        ])
    toss = "It is flipped once." if n == 1 else f"It is flipped {n} times independently."
    return rng.choice([
        f"A biased coin lands heads with probability {b}. {toss}",
        f"A weighted coin comes up heads with probability {b} on each flip. {toss}",
    ])


def gen_coin_binary(rng):
    pct = rng.randint(5, 95)
    b = Fraction(pct, 100)
    n = rng.randint(1, 4)
    lang = pick_lang(rng)
    state = coin_state(rng, lang, pct, n)
    params = {"kind": "coin", "pct": pct, "n": n}
    if n == 1:
        kind = "heads"
    else:
        kind = rng.choice(["all", "at_least", "exactly"])
    if kind == "heads":
        p = b
        cands = ["正面", "反面"] if lang == "zh" else ["Heads", "Tails"]
        prop = "掷出的是正面。" if lang == "zh" else "The flip comes up heads."
        params["event"] = ["heads"]
    elif kind == "all":
        p = b ** n
        if lang == "zh":
            cands = ["全是正面", "非全正面"]
            prop = f"{n} 次投掷全部是正面。"
        else:
            cands = [f"all {n} heads", "not all heads"]
            prop = f"All {n} flips come up heads."
        params["event"] = ["all"]
    elif kind == "at_least":
        p = 1 - (1 - b) ** n
        if lang == "zh":
            cands = ["至少一次正面", "全是反面"]
            prop = f"{n} 次投掷中至少出现一次正面。"
        else:
            cands = ["at least one heads", "all tails"]
            prop = f"At least one of the {n} flips is heads."
        params["event"] = ["at_least"]
    else:
        k = rng.randint(0, n)
        from math import comb
        p = Fraction(comb(n, k) * pct ** k * (100 - pct) ** (n - k), 100 ** n)
        if lang == "zh":
            cands = [f"恰好 {k} 次正面", "其他次数"]
            prop = f"{n} 次投掷中恰好出现 {k} 次正面。"
        else:
            cands = [f"exactly {k} heads", "any other count"]
            prop = f"Exactly {k} of the {n} flips are heads."
        params["event"] = ["exactly", k]
    return mk_pair(rng, "coin", state, None, cands, prop, p, "simple", lang, params)


def gen_coin_multi(rng):
    from math import comb
    pct = rng.randint(5, 95)
    n = rng.randint(2, 4)
    lang = pick_lang(rng)
    state = coin_state(rng, lang, pct, n)
    if lang == "zh":
        cands = [f"{k} 次正面" for k in range(n + 1)]
        instr = rng.choice(["正面出现的次数是几次？", f"{n} 次投掷中会有几次正面？"])
    else:
        cands = [f"{k} heads" for k in range(n + 1)]
        instr = rng.choice(["How many flips come up heads?", "What is the number of heads?"])
    dist = {cands[k]: Fraction(comb(n, k) * pct ** k * (100 - pct) ** (n - k), 100 ** n)
            for k in range(n + 1)}
    params = {"kind": "coin", "pct": pct, "n": n, "event": ["dist"]}
    return mk_single("coin", "choice", state, instr, cands, dist, "simple", lang, params)


# ---------------------------------------------------------------- spinner

def spinner_angles(rng, s):
    while True:
        cuts = sorted(rng.sample(range(15, 356, 5), s - 1))
        pts = [0] + cuts + [360]
        gaps = [pts[i + 1] - pts[i] for i in range(s)]
        if min(gaps) >= 15 and len(set(gaps)) > 1:
            return gaps


def spinner_state(rng, lang, cis, angles):
    s = len(angles)
    if lang == "zh":
        p1 = join_zh([f"{cname(ci, lang)}色 {a}°" for ci, a in zip(cis, angles)])
        return rng.choice([
            f"一个转盘被分成 {s} 个扇区：{p1}。指针随机停下，停在各扇区的概率与圆心角大小成正比。",
            f"圆形转盘被划分为 {s} 个区域，圆心角分别为 {p1}。拨动指针，停下时指向某区域的概率正比于该区域的角度。",
            f"转盘上有 {s} 个扇形区域（{p1}），指针等可能地停在任何角度上。",
            f"一个幸运转盘包含 {s} 个扇区：{p1}。转动一次，观察指针停在哪个扇区。",
        ])
    p2 = ", ".join([f"{cname(ci, lang)}: {a}°" for ci, a in zip(cis, angles)])
    return rng.choice([
        f"A spinner is divided into {s} sectors: {p2}. The pointer stops at uniformly random angles, "
        f"so each sector's probability is proportional to its angle.",
        f"A wheel has {s} sectors with central angles {p2}. One spin; the pointer is equally likely "
        f"to stop at any angle.",
    ])


def gen_spinner_binary(rng):
    s = rng.randint(2, 6)
    cis = rng.sample(range(len(COLORS)), s)
    angles = spinner_angles(rng, s)
    lang = pick_lang(rng)
    state = spinner_state(rng, lang, cis, angles)
    i = rng.randrange(s)
    c = cname(cis[i], lang)
    p = Fraction(angles[i], 360)
    if lang == "zh":
        cands = [f"{c}色扇区", "其他扇区"]
        prop = rng.choice([f"指针停在{c}色扇区。", f"指针最终指向{c}色区域。"])
    else:
        cands = [f"the {c} sector", "another sector"]
        prop = rng.choice([f"The pointer stops on the {c} sector.",
                           f"The pointer ends up in the {c} sector."])
    params = {"kind": "spinner", "angles": angles, "event": ["sector", i]}
    return mk_pair(rng, "spinner", state, None, cands, prop, p, "simple", lang, params)


def gen_spinner_multi(rng):
    s = rng.randint(2, 6)
    cis = rng.sample(range(len(COLORS)), s)
    angles = spinner_angles(rng, s)
    lang = pick_lang(rng)
    state = spinner_state(rng, lang, cis, angles)
    if lang == "zh":
        cands = [f"{cname(ci, lang)}色" for ci in cis]
        instr = rng.choice(["指针会停在哪个扇区？", "指针指向哪种颜色？"])
    else:
        cands = [cname(ci, lang) for ci in cis]
        instr = rng.choice(["Which sector will the pointer stop on?",
                            "Where does the pointer land?"])
    dist = {cands[i]: Fraction(angles[i], 360) for i in range(s)}
    params = {"kind": "spinner", "angles": angles, "event": ["dist"]}
    return mk_single("spinner", "choice", state, instr, cands, dist, "simple", lang, params)


# ---------------------------------------------------------------- card (OOD only)

def make_deck(rng):
    """Returns {suit_en: [ranks]}; standard 52 or a reduced custom deck."""
    if rng.random() < 0.4:
        return {en: list(RANKS) for _, en, _ in SUITS}
    suits = rng.sample(SUITS, rng.randint(2, 4))
    nranks = rng.randint(5, 13)
    ranks = rng.sample(RANKS, nranks)
    ranks.sort(key=RANKS.index)
    return {en: list(ranks) for _, en, _ in suits}


def suit_zh(en):
    return next(z for z, e, _ in SUITS if e == en)


def card_state(rng, lang, deck):
    total = sum(len(v) for v in deck.values())
    if lang == "zh":
        if set(deck) == {e for _, e, _ in SUITS} and all(len(v) == 13 for v in deck.values()):
            body = rng.choice([
                "一副标准扑克牌，共 52 张：四种花色（红桃、黑桃、方块、梅花）各 13 张（A 到 K）。",
                "一副完整的 52 张扑克牌，红桃、黑桃、方块、梅花各 A 至 K 共 13 张。",
            ])
        else:
            parts = "、".join(
                f"{suit_zh(en)} {len(ranks)} 张（{'、'.join(ranks)}）"
                for en, ranks in deck.items())
            body = rng.choice([
                f"一副自定义牌堆：{parts}，共 {total} 张。",
                f"有一副缩水牌堆，包含{parts}，总计 {total} 张牌。",
            ])
        return body + "洗匀后随机抽出一张。"
    if len(deck) == 4 and all(len(v) == 13 for v in deck.values()):
        body = rng.choice([
            "A standard 52-card deck: four suits (hearts, spades, diamonds, clubs), "
            "13 ranks each (A through K).",
            "A full poker deck of 52 cards, suits hearts/spades/diamonds/clubs, ranks A to K.",
        ])
    else:
        parts = ", ".join(f"{len(ranks)} {en} ({', '.join(ranks)})"
                          for en, ranks in deck.items())
        body = f"A custom deck: {parts}, {total} cards in total."
    return body + " One card is drawn at random after shuffling."


def gen_card_binary(rng):
    deck = make_deck(rng)
    total = sum(len(v) for v in deck.values())
    lang = pick_lang(rng)
    state = card_state(rng, lang, deck)
    has_red = any(e in RED_SUITS for e in deck)
    has_black = any(e not in RED_SUITS for e in deck)
    kinds = ["suit", "rank"] + (["color"] if has_red and has_black else [])
    kind = rng.choice(kinds)
    params = {"kind": "card", "deck": deck}
    if kind == "color":
        red = rng.random() < 0.5
        n = sum(len(v) for e, v in deck.items() if (e in RED_SUITS) == red)
        p = Fraction(n, total)
        if lang == "zh":
            cands = ["红色", "黑色"] if red else ["黑色", "红色"]
            prop = "抽到的牌是红色的。" if red else "抽到的牌是黑色的。"
        else:
            cands = ["red", "black"] if red else ["black", "red"]
            prop = "The drawn card is red." if red else "The drawn card is black."
        params["event"] = ["color", "red" if red else "black"]
    elif kind == "suit":
        en = rng.choice(list(deck))
        p = Fraction(len(deck[en]), total)
        if lang == "zh":
            s = suit_zh(en)
            cands = [s, f"非{s}"]
            prop = f"抽到的牌是{s}。"
        else:
            cands = [en, f"not {en}"]
            prop = f"The drawn card is a {en}."
        params["event"] = ["suit", en]
    else:
        rank = rng.choice([r for ranks in deck.values() for r in ranks])
        n = sum(1 for ranks in deck.values() if rank in ranks)
        p = Fraction(n, total)
        if lang == "zh":
            cands = [f"点数 {rank}", f"非 {rank}"]
            prop = f"抽到的牌点数是 {rank}。"
        else:
            cands = [f"rank {rank}", "another rank"]
            prop = f"The drawn card has rank {rank}."
        params["event"] = ["rank", rank]
    return mk_pair(rng, "card", state, None, cands, prop, p, "simple", lang, params)


def gen_card_multi(rng):
    deck = make_deck(rng)
    total = sum(len(v) for v in deck.values())
    lang = pick_lang(rng)
    state = card_state(rng, lang, deck)
    suits = list(deck)
    if lang == "zh":
        cands = [suit_zh(e) for e in suits]
        instr = rng.choice(["抽到的牌是什么花色？", "这张牌属于哪个花色？"])
    else:
        cands = list(suits)
        instr = rng.choice(["What suit is the drawn card?", "Which suit comes out?"])
    dist = {c: Fraction(len(deck[e]), total) for c, e in zip(cands, suits)}
    params = {"kind": "card", "deck": deck, "event": ["dist"]}
    return mk_single("card", "choice", state, instr, cands, dist, "simple", lang, params)


# ---------------------------------------------------------------- dice_sum (OOD only, enumerative)

def dice_sum_counts(faces):
    counts = {0: 1}
    for F in (faces if isinstance(faces, list) else [faces]):
        nxt = {}
        for s, c in counts.items():
            for face in range(1, F + 1):
                nxt[s + face] = nxt.get(s + face, 0) + c
        counts = nxt
    return counts


def dice_sum_state(rng, lang, d, F):
    if lang == "zh":
        return rng.choice([
            f"{d} 颗均匀的 {F} 面骰子（面编号 1 到 {F}）同时掷出，观察点数之和。",
            f"同时投掷 {d} 个公平的 {F} 面骰（点数 1–{F}），计算总点数。",
            f"把 {d} 枚标准 {F} 面骰子一起掷出，关心它们点数相加的结果。",
        ])
    return rng.choice([
        f"{d} fair {F}-sided dice (faces 1–{F}) are rolled together; consider the sum of the faces.",
        f"Someone rolls {d} balanced {F}-sided dice at once and adds up the results.",
    ])


def gen_dice_sum_binary(rng):
    d = rng.choice([2, 2, 3, 3, 4])
    F = rng.choice([4, 6, 8, 10, 12, 20] if d == 2 else
                   [4, 6, 8, 12] if d == 3 else [4, 6, 8])
    counts = dice_sum_counts([F] * d)
    total = F ** d
    lo, hi = d, d * F
    kind = rng.choice(["eq", "ge"])
    for _ in range(30):
        if kind == "eq":
            t = rng.randint(lo + 1, hi - 1)
            fav = counts.get(t, 0)
        else:
            t = rng.randint(lo + 2, hi - 1)
            fav = sum(c for s, c in counts.items() if s >= t)
        if 0 < fav < total:
            break
    else:
        return None
    p = Fraction(fav, total)
    lang = pick_lang(rng)
    state = dice_sum_state(rng, lang, d, F)
    if kind == "eq":
        if lang == "zh":
            cands = [f"等于 {t}", f"不等于 {t}"]
            prop = f"总点数恰好等于 {t}。"
        else:
            cands = [f"exactly {t}", f"not {t}"]
            prop = f"The sum is exactly {t}."
        params = {"kind": "dice_sum", "d": d, "faces": F, "event": ["eq", t]}
    else:
        if lang == "zh":
            cands = [f"至少 {t}", f"小于 {t}"]
            prop = f"总点数大于等于 {t}。"
        else:
            cands = [f"at least {t}", f"less than {t}"]
            prop = f"The sum is at least {t}."
        params = {"kind": "dice_sum", "d": d, "faces": F, "event": ["ge", t]}
    return mk_pair(rng, "dice_sum", state, None, cands, prop, p, "enumerative", lang, params)


def gen_dice_sum_multi(rng):
    if rng.random() < 0.5:
        faces = sorted(rng.sample([4, 6, 8, 10, 12, 20], 2))
    else:
        faces = [rng.choice([4, 6, 8])] * 3
    counts = dice_sum_counts(faces)
    total = 1
    for F in faces:
        total *= F
    lang = pick_lang(rng)
    d = len(faces)
    if len(set(faces)) == 1:
        state = dice_sum_state(rng, lang, d, faces[0])
    elif lang == "zh":
        state = rng.choice([
            f"一颗 {faces[0]} 面骰和一颗 {faces[1]} 面骰（均均匀，面编号从 1 开始）同时掷出，观察点数之和。",
            f"同时投掷一个公平的 {faces[0]} 面骰和一个公平的 {faces[1]} 面骰，计算总点数。",
        ])
    else:
        state = rng.choice([
            f"A fair {faces[0]}-sided die and a fair {faces[1]}-sided die are rolled together; "
            f"consider the sum of the faces.",
            f"Someone rolls one balanced {faces[0]}-sided die and one balanced {faces[1]}-sided die "
            f"and adds up the results.",
        ])
    lo, hi = d, sum(faces)
    sums = list(range(lo, hi + 1))
    zh_style = rng.random() < 0.5
    if lang == "zh":
        cands = [f"{s} 点" if zh_style else str(s) for s in sums]
        instr = rng.choice(["总点数是多少？", "两颗骰子的点数之和是多少？", "点数之和等于几？"])
    else:
        cands = [str(s) for s in sums]
        instr = rng.choice(["What is the sum of the dice?", "What total do the dice show?"])
    dist = {c: Fraction(counts.get(s, 0), total) for c, s in zip(cands, sums)}
    params = {"kind": "dice_sum", "d": d, "faces": faces, "event": ["dist"]}
    return mk_single("dice_sum", "choice", state, instr, cands, dist,
                     "enumerative", lang, params)


# ---------------------------------------------------------------- family B: deterministic

WORDS = {
    "zh": ["苹果", "香蕉", "橘子", "葡萄", "西瓜", "桃子", "铅笔", "橡皮",
           "桌子", "椅子", "书包", "台灯", "猫", "狗", "鸟", "鱼"],
    "en": ["apple", "banana", "grape", "pencil", "chair", "lamp", "cat", "dog",
           "river", "stone", "cloud", "train", "bottle", "mirror"],
}


def one_hot(cands, idx):
    return {c: Fraction(1 if i == idx else 0) for i, c in enumerate(cands)}


def gen_arithmetic(rng):
    lang = pick_lang(rng)
    mag = rng.choice([(10, 99), (100, 999), (1000, 9999)])
    if rng.random() < 0.7:
        a, b = rng.randint(*mag), rng.randint(*mag)
        op = rng.choice(["+", "-", "×"])
        val = {"+": a + b, "-": a - b, "×": a * b}[op]
        expr = f"{a} {op} {b}"
        bucket = f"two{op}{mag[1]}"
    else:
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 20)
        val = (a + b) * c
        expr = f"({a} + {b}) × {c}"
        bucket = "three"
    state = ("请计算下面这个整数算式。" if lang == "zh" else
             "Compute the following integer expression.")
    params = {"kind": "arithmetic", "expr": expr, "answer": val}
    if rng.random() < 0.5:
        distract = {val + d for d in (1, -1, 2, -2, 10, -10, 100, -100)} - {val}
        ds = rng.sample(sorted(distract), 3)
        cands = [str(v) for v in [val] + ds]
        rng.shuffle(cands)
        idx = cands.index(str(val))
        instr = (rng.choice([f"{expr} 等于多少？", f"计算：{expr} = ?"])
                 if lang == "zh" else
                 rng.choice([f"What is {expr}?", f"Evaluate: {expr}"]))
        return mk_single("arithmetic", "choice", state, instr, cands,
                         one_hot(cands, idx), "simple", lang, params), bucket
    correct = rng.random() < 0.5
    v = val if correct else val + rng.choice([-10, -2, -1, 1, 2, 10])
    prop = (rng.choice([f"{expr} 等于 {v}。", f"{expr} 的结果是 {v}。"])
            if lang == "zh" else
            rng.choice([f"{expr} equals {v}.", f"The value of {expr} is {v}."]))
    return mk_single("arithmetic", "noul", state, prop, ["yes", "no"],
                     {"yes": Fraction(int(correct)),
                      "no": Fraction(int(not correct))},
                     "simple", lang, params), bucket


def gen_comparison(rng):
    lang = pick_lang(rng)
    kind = rng.choice(["int", "dec", "frac"])
    while True:
        if kind == "int":
            v1, v2 = Fraction(rng.randint(-10 ** 6, 10 ** 6)), Fraction(rng.randint(-10 ** 6, 10 ** 6))
            s1, s2 = str(v1), str(v2)
        elif kind == "dec":
            dp = rng.randint(1, 3)
            v1 = Fraction(rng.randint(-10 ** 5, 10 ** 5), 10 ** dp)
            v2 = Fraction(rng.randint(-10 ** 5, 10 ** 5), 10 ** dp)
            s1, s2 = f"{float(v1):.{dp}f}", f"{float(v2):.{dp}f}"
        else:
            v1 = Fraction(rng.randint(1, 60), rng.randint(2, 60))
            v2 = Fraction(rng.randint(1, 60), rng.randint(2, 60))
            s1, s2 = str(v1), str(v2)
        if v1 != v2 and s1 != s2:
            break
    state = ("比较两个数的大小。" if lang == "zh" else
             "Compare the magnitudes of two numbers.")
    params = {"kind": "comparison", "a": s1, "b": s2,
              "a_value": [v1.numerator, v1.denominator],
              "b_value": [v2.numerator, v2.denominator]}
    if rng.random() < 0.5:
        cands = [s1, s2]
        rng.shuffle(cands)
        idx = cands.index(s1 if v1 > v2 else s2)
        instr = ("哪个数更大？" if lang == "zh" else "Which number is larger?")
        return mk_single("comparison", "choice", state, instr, cands,
                         one_hot(cands, idx), "simple", lang, params), kind
    prop = (f"{s1} 大于 {s2}。" if lang == "zh" else f"{s1} is greater than {s2}.")
    yes = v1 > v2
    return mk_single("comparison", "noul", state, prop, ["yes", "no"],
                     {"yes": Fraction(int(yes)), "no": Fraction(int(not yes))},
                     "simple", lang, params), kind


def gen_membership(rng):
    lang = pick_lang(rng)
    kind = rng.choice(["word", "number"])
    pool = WORDS[lang] if kind == "word" else [str(i) for i in range(10, 999)]
    lst = rng.sample(pool, rng.randint(5, 8))
    member = rng.random() < 0.5
    if member:
        q = rng.choice(lst)
    else:
        q = rng.choice([x for x in pool if x not in lst])
    lst_str = "、".join(lst) if lang == "zh" else ", ".join(lst)
    state = (f"给定列表：[{lst_str}]。" if lang == "zh"
             else f"Given the list: [{lst_str}].")
    params = {"kind": "membership", "list": lst, "query": q}
    if rng.random() < 0.6:
        prop = (rng.choice([f"“{q}” 在该列表中。", f"{q} 是列表中的元素。"])
                if lang == "zh" else
                rng.choice([f'"{q}" is in the list.', f"{q} is an element of the list."]))
        return mk_single("membership", "noul", state, prop, ["yes", "no"],
                         {"yes": Fraction(int(member)),
                          "no": Fraction(int(not member))},
                         "simple", lang, params), kind
    others = rng.sample([x for x in pool if x not in lst], 3)
    cands = [rng.choice(lst)] + others
    rng.shuffle(cands)
    idx = 0
    for i, c in enumerate(cands):
        if c in lst:
            idx = i
            break
    instr = ("以下哪个元素在列表中？" if lang == "zh" else
             "Which of the following is in the list?")
    return mk_single("membership", "choice", state, instr, cands,
                     one_hot(cands, idx), "simple", lang, params), kind


def gen_date(rng):
    lang = pick_lang(rng)
    base = date(rng.randint(2024, 2027), rng.randint(1, 12), rng.randint(1, 28))
    k = rng.randint(1, 365)
    res = base + timedelta(days=k)
    params = {"kind": "date", "base": base.isoformat(), "days": k,
              "answer": res.isoformat()}
    fmt = (lambda d: f"{d.year}年{d.month}月{d.day}日") if lang == "zh" else date.isoformat
    distract = {res + timedelta(days=d) for d in (-7, -3, -1, 1, 2, 3, 7)} - {res}
    ds = rng.sample(sorted(distract), 3)
    cands = [fmt(d) for d in [res] + ds]
    rng.shuffle(cands)
    idx = cands.index(fmt(res))
    state = ("推算日期。" if lang == "zh" else "Do the date arithmetic.")
    instr = (rng.choice([f"{fmt(base)}加 {k} 天是哪一天？", f"{fmt(base)}之后第 {k} 天是哪天？"])
             if lang == "zh" else
             rng.choice([f"What date is {k} days after {fmt(base)}?",
                         f"What is the date {k} days past {fmt(base)}?"]))
    return mk_single("date", "choice", state, instr, cands,
                     one_hot(cands, idx), "simple", lang, params), f"k{k // 30}"


def gen_string(rng):
    import string as _s
    lang = pick_lang(rng)
    kind = rng.choice(["length", "palindrome"])
    alpha = _s.ascii_letters + _s.digits
    if kind == "length":
        L = rng.randint(3, 12)
        stext = "".join(rng.choice(alpha) for _ in range(L))
        distract = {L + d for d in (-2, -1, 1, 2, 3)} - {L}
        ds = [d for d in rng.sample(sorted(distract), 4) if d > 0][:3]
        cands = [str(v) for v in [L] + ds]
        rng.shuffle(cands)
        idx = cands.index(str(L))
        state = ("观察下面这个字符串。" if lang == "zh" else
                 "Consider the following string.")
        instr = (f"字符串 “{stext}” 的长度（字符数）是多少？" if lang == "zh" else
                 f'What is the length (number of characters) of the string "{stext}"?')
        params = {"kind": "string", "s": stext, "task": "length"}
        return mk_single("string", "choice", state, instr, cands,
                         one_hot(cands, idx), "simple", lang, params), kind
    half_len = rng.randint(2, 5)
    if rng.random() < 0.5:
        half = "".join(rng.choice(alpha) for _ in range(half_len))
        mid = rng.choice(alpha) if rng.random() < 0.5 else ""
        stext = half + mid + half[::-1]
        is_pal = True
    else:
        while True:
            stext = "".join(rng.choice(alpha) for _ in range(2 * half_len))
            if stext != stext[::-1]:
                break
        is_pal = False
    state = ("判断字符串的性质。" if lang == "zh" else
             "Judge a property of the string.")
    prop = (f"字符串 “{stext}” 是回文（正读反读相同）。" if lang == "zh" else
            f'The string "{stext}" is a palindrome (reads the same forwards and backwards).')
    params = {"kind": "string", "s": stext, "task": "palindrome"}
    return mk_single("string", "noul", state, prop, ["yes", "no"],
                     {"yes": Fraction(int(is_pal)), "no": Fraction(int(not is_pal))},
                     "simple", lang, params), kind


B_GENS = {"arithmetic": gen_arithmetic, "comparison": gen_comparison,
          "membership": gen_membership, "date": gen_date, "string": gen_string}
A_BIN_GENS = {"urn": gen_urn_binary, "lottery": gen_lottery_binary,
              "die": gen_die_binary, "coin": gen_coin_binary,
              "spinner": gen_spinner_binary, "card": gen_card_binary,
              "dice_sum": gen_dice_sum_binary}
A_MUL_GENS = {"urn": gen_urn_multi, "lottery": gen_lottery_multi,
              "die": gen_die_multi, "coin": gen_coin_multi,
              "spinner": gen_spinner_multi, "card": gen_card_multi,
              "dice_sum": gen_dice_sum_multi}


# ---------------------------------------------------------------- driver

def bucket_of(ev):
    if ev.get("bucket") is not None:
        return ev["bucket"]
    p = ev["recs"][0]["meta"]["params"]
    k = p["kind"]
    if k == "urn":
        return f"{len(p['colors'])}c_d{p['draws']}{'r' if p['replace'] else 'n'}"
    if k == "lottery":
        return f"k{p['K'] // 10}"
    if k == "die":
        return f"F{p['faces']}_{'l' if p['loaded'] else 'f'}"
    if k == "coin":
        return f"b{p['pct'] // 10}_n{p['n']}"
    if k == "spinner":
        return f"s{len(p['angles'])}"
    if k == "card":
        deck = p["deck"]
        std = len(deck) == 4 and all(len(v) == 13 for v in deck.values())
        return "std" if std else f"{len(deck)}x{len(next(iter(deck.values())))}"
    if k == "dice_sum":
        return f"d{p['d']}F{p['faces']}"
    return "b"


def text_hash(r):
    q = r["instruction"] or r["proposition"]
    blob = r["state"] + "|" + q + "|" + "\x00".join(r["candidates"])
    return hashlib.sha1(blob.encode()).hexdigest()


def assign_splits(events, rng):
    strata = {}
    for ev in events:
        ood = ev["mech"] in OOD_MECHS
        key = (ev["mech"], ev["recs"][0]["difficulty"], bucket_of(ev), ood)
        strata.setdefault(key, []).append(ev)
    for (mech, _diff, _bucket, ood), grp in strata.items():
        rng.shuffle(grp)
        n = len(grp)
        if ood:
            splits = ["ood"] * n
        elif n < 10:
            splits = ["train"] * n
        else:
            ntr, ndev, ncal = n * 7 // 10, n // 10, n // 10
            splits = (["train"] * ntr + ["dev"] * ndev + ["calibration"] * ncal
                      + ["test"] * (n - ntr - ndev - ncal))
        for ev, sp in zip(grp, splits):
            for r in ev["recs"]:
                r["split"] = sp


def float_dist(dist):
    vals = {k: float(v) for k, v in dist.items()}
    if vals:
        kmax = max(vals, key=vals.get)
        vals[kmax] += 1.0 - sum(vals.values())
    return vals


def build(n_a=60000, n_b=30000, seed=20260921):
    rng = random.Random(seed)
    seen = set()
    counters = {}
    events = []

    def fresh_id(mech):
        counters[mech] = counters.get(mech, 0) + 1
        return f"{mech}_{counters[mech]:06d}"

    def try_add(gen, takes_bucket=False):
        for _ in range(50):
            out = gen(rng)
            if out is None:
                continue
            ev, bucket = out if takes_bucket else (out, None)
            hs = [text_hash(r) for r in ev["recs"]]
            if any(h in seen for h in hs):
                continue
            seen.update(hs)
            ids = [fresh_id(ev["mech"]) for _ in ev["recs"]]
            for r, i in zip(ev["recs"], ids):
                r["id"] = i
                r.setdefault("paired_id", None)
            if ev["pair"]:
                ev["recs"][0]["paired_id"] = ids[1]
                ev["recs"][1]["paired_id"] = ids[0]
            if bucket is not None:
                ev["bucket"] = bucket
            events.append(ev)
            return True
        return False

    scale = n_a / 60000.0
    for mech, q in A_BIN_Q.items():
        for _ in range(max(1, round(q * scale))):
            try_add(A_BIN_GENS[mech])
    for mech, q in A_MUL_Q.items():
        for _ in range(max(1, round(q * scale))):
            try_add(A_MUL_GENS[mech])
    b_start = len(events)
    for mech, q in B_Q.items():
        for _ in range(max(1, round(q * n_b / 30000.0))):
            try_add(B_GENS[mech], takes_bucket=True)

    assign_splits(events[:b_start], rng)
    assign_splits(events[b_start:], rng)

    records = []
    for ev in events:
        for r in ev["recs"]:
            records.append({
                "id": r["id"],
                "family": "deterministic" if ev["mech"] in B_Q else "random_mechanism",
                "mechanism": r["mechanism"],
                "primitive": r["primitive"],
                "state": r["state"],
                "instruction": r["instruction"],
                "proposition": r["proposition"],
                "candidates": r["candidates"],
                "target_distribution": float_dist(r["target"]),
                "difficulty": r["difficulty"],
                "split": r["split"],
                "paired_id": r["paired_id"],
                "meta": r["meta"],
            })
    order = {s: i for i, s in enumerate(SPLIT_ORDER)}
    records.sort(key=lambda r: (order[r["split"]], r["id"]))
    return records


def summarize(records):
    from collections import Counter
    by_split = Counter(r["split"] for r in records)
    by_mech = Counter((r["split"], r["mechanism"]) for r in records)
    by_prim = Counter(r["primitive"] for r in records)
    paired = sum(1 for r in records if r["paired_id"])
    zh = sum(1 for r in records if r["meta"]["lang"] == "zh")
    lines = [f"total: {len(records)}"]
    lines += [f"  {s}: {by_split.get(s, 0)}" for s in SPLIT_ORDER]
    lines.append(f"primitive: {dict(by_prim)}")
    lines.append(f"paired records: {paired}  zh ratio: {zh / len(records):.3f}")
    for s in SPLIT_ORDER:
        ms = {m: c for (sp, m), c in sorted(by_mech.items()) if sp == s}
        if ms:
            lines.append(f"[{s}] {ms}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--n", type=int, default=90000, help="total items (A:B = 2:1)")
    ap.add_argument("--n-a", type=int, default=None)
    ap.add_argument("--n-b", type=int, default=None)
    args = ap.parse_args()
    n_a = args.n_a if args.n_a is not None else round(args.n * 2 / 3)
    n_b = args.n_b if args.n_b is not None else args.n - n_a
    records = build(n_a=n_a, n_b=n_b, seed=args.seed)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(summarize(records))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
