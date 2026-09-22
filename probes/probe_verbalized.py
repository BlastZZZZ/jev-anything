"""Verbalized-probability expansion: is the perfect calibration a fluke?

Part 1: 10 binary events with analytic probabilities (dice, cards, urns,
compound events). Model generates a percentage number at T=0.
Part 2: K-way full distributions (K=2..6) generated as JSON in one shot,
including non-uniform and near-deterministic mechanisms.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import httpx
from systemone import BASE_URL, JOURNAL, MODEL

REPO_ROOT = Path(__file__).resolve().parents[1]


def journal(rec):
    Path(JOURNAL).parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def verbalized_p(state, proposition, tag):
    prompt = (
        "Read the state, then estimate the probability that the statement is "
        "true. Think about the underlying random mechanism.\n\n"
        f"STATE:\n{state}\n\nSTATEMENT:\n{proposition}\n\n"
        "Answer with only a number between 0 and 100 (the percentage)."
    )
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0.0}
    text = httpx.post(f"{BASE_URL}/chat/completions", json=body,
                      timeout=60).json()["choices"][0]["message"]["content"]
    journal({"tag": tag, "method": "verbalized", "raw_text": text.strip()})
    digits = "".join(c for c in text if c.isdigit() or c == ".")
    return float(digits) / 100.0


def verbalized_dist(state, instruction, candidates, tag):
    spec = ", ".join(f'"{c}"' for c in candidates)
    prompt = (
        "Read the state, then assign a probability to every possible answer. "
        "Think about the underlying random mechanism.\n\n"
        f"STATE:\n{state}\n\nQUESTION:\n{instruction}\n\n"
        f"POSSIBLE ANSWERS: {spec}\n\n"
        'Return a JSON object mapping each answer to its probability, e.g. '
        '{"answer": 0.5}. Probabilities must sum to 1. Return only JSON.'
    )
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0.0,
            "response_format": {"type": "json_object"}}
    text = httpx.post(f"{BASE_URL}/chat/completions", json=body,
                      timeout=60).json()["choices"][0]["message"]["content"]
    journal({"tag": tag, "method": "verbalized_dist", "raw_text": text})
    return json.loads(text)


BINARY = [
    # (name, state, proposition, reference)
    ("die_six",
     "A fair six-sided die was rolled in a sealed box. The result is hidden.",
     "The die shows a six.", 1 / 6),
    ("die_even",
     "A fair six-sided die was rolled in a sealed box. The result is hidden.",
     "The die shows an even number.", 0.5),
    ("dice_sum7",
     "Two fair six-sided dice were rolled in a sealed box. The results are hidden.",
     "The two dice sum to exactly 7.", 6 / 36),
    ("dice_sum2",
     "Two fair six-sided dice were rolled in a sealed box. The results are hidden.",
     "The two dice sum to exactly 2.", 1 / 36),
    ("card_hearts",
     "A card was drawn uniformly at random from a standard 52-card deck. It is face down.",
     "The card is a heart.", 0.25),
    ("card_ace",
     "A card was drawn uniformly at random from a standard 52-card deck. It is face down.",
     "The card is an ace.", 1 / 13),
    ("urn_green",
     "An urn contains 7 red, 2 blue and 1 green ball. One ball was drawn "
     "uniformly at random and is hidden in a closed hand.",
     "The drawn ball is green.", 0.1),
    ("two_coins_at_least_one",
     "Two fair coins were flipped in a sealed room. Both results are hidden.",
     "At least one of the two coins landed heads.", 0.75),
    ("lottery20_ball17",
     "A fair lottery machine contains exactly 20 identical balls, numbered 1 "
     "through 20. One ball was drawn uniformly at random; the result is hidden.",
     "The drawn ball is ball number 17.", 0.05),
    ("biased70_both",
     "A biased coin that lands heads with probability 0.7 was flipped twice in "
     "a sealed room. Both results are hidden.",
     "Both flips landed heads.", 0.49),
]

DIST_CASES = [
    ("dist_lottery5",
     "A fair lottery machine contains exactly 5 identical balls, numbered 1 "
     "through 5. One ball was drawn uniformly at random; the result is hidden.",
     "Which ball was drawn?",
     [f"Ball number {i}" for i in range(1, 6)],
     [0.2] * 5),
    ("dist_urn",
     "An urn contains 7 red, 2 blue and 1 green ball. One ball was drawn "
     "uniformly at random and is hidden in a closed hand.",
     "What color is the drawn ball?",
     ["Red", "Blue", "Green"],
     [0.7, 0.2, 0.1]),
    ("dist_die6",
     "A fair six-sided die was rolled in a sealed box. The result is hidden.",
     "What number does the die show?",
     [str(i) for i in range(1, 7)],
     [1 / 6] * 6),
    ("dist_loaded_die",
     "A loaded six-sided die lands on 6 with probability 0.5 and on each of "
     "1, 2, 3, 4, 5 with probability 0.1. It was rolled in a sealed box; the "
     "result is hidden.",
     "What number does the die show?",
     [str(i) for i in range(1, 7)],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
    ("dist_arith",
     "A spreadsheet cell computes the integer sum 2 + 2 and displays the result.",
     "What value does the cell display?",
     ["4", "5"],
     [1.0, 0.0]),
]


def tv(dist, candidates, ref):
    return 0.5 * sum(abs(dist.get(c, 0.0) - r) for c, r in zip(candidates, ref))



if __name__ == "__main__":
    rows = []
    print("=== Part 1: binary verbalized ===")
    for name, state, prop, ref in BINARY:
        for rep in range(2):
            p = verbalized_p(state, prop, f"{name}/rep{rep}")
            rows.append((name, rep, ref, p))
            print(f"{name:22s} rep{rep}  ref={ref:.4f}  got={p:.4f}  err={abs(p-ref):.4f}")

    print("\n=== Part 2: K-way verbalized distributions ===")
    dist_rows = []
    for name, state, instr, cands, ref in DIST_CASES:
        for rep in range(2):
            try:
                d = verbalized_dist(state, instr, cands, f"{name}/rep{rep}")
                s = sum(d.values())
                norm = {k: v / s for k, v in d.items()}
                err = tv(norm, cands, ref)
                dist_rows.append((name, rep, ref, norm, err))
                short = {k: round(v, 3) for k, v in norm.items()}
                print(f"{name:18s} rep{rep}  TV={err:.3f}  {short}")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                dist_rows.append((name, rep, ref, None, None))
                print(f"{name:18s} rep{rep}  PARSE FAILURE: {e}")

    out = REPO_ROOT / "results" / "verbalized_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"binary": rows,
                   "distributions": dist_rows}, f, ensure_ascii=False, indent=1, default=str)
    print(f"-> {out}")

    errs = [abs(p - r) for _, _, r, p in rows]
    print(f"\nbinary: mean |err| = {sum(errs)/len(errs):.4f} over {len(errs)} trials")
    tvs = [e for *_, e in dist_rows if e is not None]
    if tvs:
        print(f"K-way:  mean TV   = {sum(tvs)/len(tvs):.4f} over {len(tvs)} trials")
