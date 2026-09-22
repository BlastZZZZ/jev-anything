"""Novel-mechanism probes: memorization or genuine calibrated reasoning?

All cases use non-round, non-textbook numbers so the answer cannot be a
memorized pattern. Same verbalized elicitation as probe_verbalized.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_verbalized import verbalized_p, verbalized_dist, tv, JOURNAL

REPO_ROOT = Path(__file__).resolve().parents[1]

BINARY = [
    ("urn13_5red",
     "An urn contains exactly 13 balls: 5 red and 8 black. One ball was drawn "
     "uniformly at random and is hidden.",
     "The drawn ball is red.", 5 / 13),
    ("lottery17_ball9",
     "A fair lottery machine contains exactly 17 identical balls, numbered 1 "
     "through 17. One ball was drawn uniformly at random; the result is hidden.",
     "The drawn ball is ball number 9.", 1 / 17),
    ("d11_seven",
     "A fair eleven-sided die (faces numbered 1 through 11) was rolled in a "
     "sealed box. The result is hidden.",
     "The die shows a seven.", 1 / 11),
    ("coin63_tails",
     "A biased coin that lands heads with probability 0.63 was flipped in a "
     "sealed room. The result is hidden.",
     "The coin landed tails.", 0.37),
    ("two_biased_both",
     "Two biased coins were flipped in a sealed room. The first lands heads "
     "with probability 0.63, the second with probability 0.41. The flips are "
     "independent. Both results are hidden.",
     "Both coins landed heads.", 0.63 * 0.41),
    ("urn23_blue",
     "An urn contains exactly 23 balls: 9 red, 8 blue and 6 green. One ball "
     "was drawn uniformly at random and is hidden.",
     "The drawn ball is blue.", 8 / 23),
    ("two_draws_no_replace",
     "A bag contains 7 red and 5 black balls. Two balls were drawn one after "
     "another without replacement, uniformly at random. Both are hidden.",
     "Both drawn balls are red.", (7 / 12) * (6 / 11)),
    ("at_least_one_6_in_4",
     "A fair six-sided die was rolled four times in a sealed box. All results "
     "are hidden.",
     "At least one of the four rolls is a six.", 1 - (5 / 6) ** 4),
    ("conditional_after_red",
     "A bag contained 3 red and 5 black balls. One ball was drawn and it was "
     "red; it was set aside. Then a second ball was drawn from the remaining "
     "balls, uniformly at random. It is hidden.",
     "The second ball is red.", 2 / 7),
    ("three_dice_sum10",
     "Three fair six-sided dice were rolled in a sealed box. The results are "
     "hidden.",
     "The three dice sum to exactly 10.", 27 / 216),
]

DIST_CASES = [
    ("dist_spinner4",
     "A spinner is divided into four sectors with unequal angles: sector North "
     "covers 37% of the circle, East 26%, South 22%, West 15%. The spinner was "
     "spun and stopped hidden from view.",
     "Which sector did the spinner stop on?",
     ["North", "East", "South", "West"],
     [0.37, 0.26, 0.22, 0.15]),
    ("dist_lottery23",
     "A fair lottery machine contains exactly 23 identical balls, numbered 1 "
     "through 23. One ball was drawn uniformly at random; the result is hidden.",
     "Which ball was drawn?",
     [f"Ball {i}" for i in range(1, 8)] + ["Some other ball (8-23)"],
     [1 / 23] * 7 + [16 / 23]),
    ("dist_loaded_d11",
     "A loaded eleven-sided die (faces 1-11) lands on 11 with probability 0.31 "
     "and on each of the other ten faces with probability 0.069. It was rolled "
     "in a sealed box; the result is hidden.",
     "What does the die show?",
     ["11", "Some face from 1 to 10"],
     [0.31, 0.69]),
    ("dist_urn23",
     "An urn contains exactly 23 balls: 9 red, 8 blue and 6 green. One ball "
     "was drawn uniformly at random and is hidden.",
     "What color is the drawn ball?",
     ["Red", "Blue", "Green"],
     [9 / 23, 8 / 23, 6 / 23]),
]

rows = []
print("=== Binary, novel mechanisms ===")
for name, state, prop, ref in BINARY:
    for rep in range(2):
        p = verbalized_p(state, prop, f"novel/{name}/rep{rep}")
        rows.append((name, rep, ref, p))
        print(f"{name:22s} rep{rep}  ref={ref:.4f}  got={p:.4f}  err={abs(p-ref):.4f}")

print("\n=== K-way, novel mechanisms ===")
dist_rows = []
for name, state, instr, cands, ref in DIST_CASES:
    for rep in range(2):
        try:
            d = verbalized_dist(state, instr, cands, f"novel/{name}/rep{rep}")
            s = sum(d.values())
            norm = {k: v / s for k, v in d.items()}
            err = tv(norm, cands, ref)
            dist_rows.append((name, rep, ref, norm, err))
            short = {k: round(v, 3) for k, v in norm.items()}
            print(f"{name:18s} rep{rep}  TV={err:.3f}  {short}")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            dist_rows.append((name, rep, ref, None, None))
            print(f"{name:18s} rep{rep}  PARSE FAILURE: {e}")

out = REPO_ROOT / "results" / "novel_summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump({"binary": rows, "distributions": dist_rows},
              f, ensure_ascii=False, indent=1, default=str)
print(f"-> {out}")

errs = [abs(p - r) for _, _, r, p in rows]
print(f"\nbinary: mean |err| = {sum(errs)/len(errs):.4f} over {len(errs)} trials")
tvs = [e for *_, e in dist_rows if e is not None]
if tvs:
    print(f"K-way:  mean TV   = {sum(tvs)/len(tvs):.4f} over {len(tvs)} trials")
