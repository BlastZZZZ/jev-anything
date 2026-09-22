"""Distribution probes with analytic ground truth (NanoJev-style).

Each case has a known reference probability distribution. We measure how
close the model's reported distribution is (TV distance), whether it is
overconfident on pure-chance events, and whether Choice and Noul agree on
the same binary event.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systemone import LABELS, choice, noul

REPO_ROOT = Path(__file__).resolve().parents[1]
REPEATS = 3


def uniform(k):
    return {LABELS[i]: 1.0 / k for i in range(k)}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def lottery_state(k):
    return (
        f"A fair lottery machine contains exactly {k} identical balls, numbered "
        f"1 through {k}. The machine mixes them thoroughly and draws one ball "
        f"uniformly at random. The draw has been made but the result is hidden."
    )


CASES = []


def add(name, kind, state, instruction=None, candidates=None,
        proposition=None, reference=None):
    CASES.append(dict(name=name, kind=kind, state=state, instruction=instruction,
                      candidates=candidates, proposition=proposition,
                      reference=reference))


# --- pure chance: uniform lotteries, K=2/5/20 ---
for k in (2, 5, 20):
    add(f"lottery_k{k}", "choice", lottery_state(k),
        instruction="Which ball was drawn?",
        candidates=[f"Ball number {i + 1}" for i in range(k)],
        reference=uniform(k))

# --- weighted lottery ---
add("lottery_weighted", "choice",
    "A lottery machine contains 10 balls: 7 red, 2 blue, 1 green. It draws one "
    "ball uniformly at random. The draw has been made but the result is hidden.",
    instruction="What color is the drawn ball?",
    candidates=["Red", "Blue", "Green"],
    reference={LABELS[0]: 0.7, LABELS[1]: 0.2, LABELS[2]: 0.1})

# --- fair coin: Choice and Noul on the SAME event (cross-primitive) ---
COIN = ("A fair coin was flipped in a sealed room. The result is covered and "
        "unknown to everyone.")
add("coin_choice", "choice", COIN,
    instruction="How did the coin land?",
    candidates=["Heads", "Tails"],
    reference=uniform(2))
add("coin_noul", "noul", COIN,
    proposition="The coin landed heads.",
    reference=0.5)

# --- biased coin 70% ---
BIASED = ("A biased coin that lands heads with probability 0.7 was flipped in a "
          "sealed room. The result is covered and unknown to everyone.")
add("coin70_choice", "choice", BIASED,
    instruction="How did the coin land?",
    candidates=["Heads", "Tails"],
    reference={LABELS[0]: 0.7, LABELS[1]: 0.3})
add("coin70_noul", "noul", BIASED,
    proposition="The coin landed heads.",
    reference=0.7)

# --- deterministic sanity check: should be confident ---
add("arith_choice", "choice",
    "A spreadsheet cell computes the integer sum 2 + 2 and displays the result.",
    instruction="What value does the cell display?",
    candidates=["4", "5"],
    reference={LABELS[0]: 1.0, LABELS[1]: 0.0})

# --- order permutation control: same lottery, reversed options ---
add("lottery_k5_reversed", "choice", lottery_state(5),
    instruction="Which ball was drawn?",
    candidates=[f"Ball number {i + 1}" for i in reversed(range(5))],
    reference={LABELS[i]: 0.2 for i in range(5)})


def run():
    results = []
    for case in CASES:
        for rep in range(REPEATS):
            tag = f"{case['name']}/rep{rep}"
            if case["kind"] == "choice":
                dist, rec = choice(case["state"], case["instruction"],
                                   case["candidates"], tag=tag)
                result = dict(case=case["name"], rep=rep, kind="choice",
                              distribution=dist,
                              tv=tv(dist, case["reference"]),
                              max_prob=max(dist.values()),
                              latency_ms=rec["latency_ms"])
            else:
                p_yes, rec = noul(case["state"], case["proposition"], tag=tag)
                result = dict(case=case["name"], rep=rep, kind="noul",
                              p_yes=p_yes, reference=case["reference"],
                              abs_err=abs(p_yes - case["reference"]),
                              latency_ms=rec["latency_ms"])
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    out = REPO_ROOT / "results" / "probe_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n{len(results)} probe results written to {out}")


if __name__ == "__main__":
    run()
