"""Reverse comparison: run NanoJev public weights on our probability test split.

choice rows -> their choice question (criteria {label: candidate_text});
noul rows -> their boolean question (false/true mapped to no/yes).
Metrics mirror evaluate.py: accuracy / NLL / Brier / ECE10 / TV, grouped by
mechanism x difficulty, plus paired choice-vs-noul consistency.

Environment variables:
  XIAOJEV_NANOJEV_REPO   local clone of https://github.com/TianyuCodings/NanoJev
  XIAOJEV_NANOJEV_CKPT   NanoJev weights path or HF id (default C-Tianyu/NanoJev)
  XIAOJEV_PROB_DATA      probability-task JSONL (default <repo>/data/train_v1.jsonl)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

import torch  # noqa: E402

from train import LABELS, load_rows  # noqa: E402
from evaluate import row_metrics, summarize, paired_consistency  # noqa: E402

NANOJEV_CKPT = os.environ.get("XIAOJEV_NANOJEV_CKPT", "C-Tianyu/NanoJev")
OUT = REPO_ROOT / "results" / "compare_reverse_nanojev.json"


def nanojev_scripts():
    env = os.environ.get("XIAOJEV_NANOJEV_REPO")
    p = Path(env) / "scripts" if env else None
    if p is None or not p.is_dir():
        raise SystemExit(
            "NanoJev scripts not found. Clone "
            "https://github.com/TianyuCodings/NanoJev and set "
            "XIAOJEV_NANOJEV_REPO to the clone."
        )
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=NANOJEV_CKPT,
                    help="NanoJev weights path or HF id (default: %(default)s)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=None,
                    help="probability JSONL (default: $XIAOJEV_PROB_DATA or <repo>/data/train_v1.jsonl)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    sys.path.insert(0, str(nanojev_scripts()))
    from predict_toy_decisions import DecisionPredictor  # noqa: E402

    rows = load_rows(args.split, args.data) if args.data else load_rows(args.split)
    predictor = DecisionPredictor(
        args.ckpt, max_length=8192,
        disable_native_triton=False)
    preds = {}
    B = 32
    for off in range(0, len(rows), B):
        chunk = rows[off : off + B]
        requests = []
        for row in chunk:
            if row["primitive"] == "choice":
                q = {"action": {"type": "choice", "instructions": row["instruction"],
                                "criteria": {LABELS[i]: c for i, c in enumerate(row["candidates"])}}}
            else:
                q = {"verdict": {"type": "boolean",
                                 "instructions": "Judge whether the following statement about the state is true.\n\nSTATEMENT:\n" + row["proposition"],
                                 "criteria": {"false": "The statement is false.", "true": "The statement is true."}}}
            requests.append({"id": row["id"], "state": row["state"], "questions": q})
        result = predictor.predict({"states": requests}, batch_questions=B)
        for state in result["states"]:
            row = next(r for r in chunk if r["id"] == state["id"])
            if row["primitive"] == "choice":
                probs = state["answers"]["action"]["probabilities"]
                preds[row["id"]] = torch.tensor([probs[LABELS[i]] for i in range(len(row["candidates"]))])
            else:
                probs = state["answers"]["verdict"]["probabilities"]
                by_cand = {"yes": probs["true"], "no": probs["false"]}
                preds[row["id"]] = torch.tensor([by_cand[c] for c in row["candidates"]])
        if (off // B) % 20 == 0:
            print(f"{off + len(chunk)}/{len(rows)}", flush=True)

    items, groups = [], defaultdict(list)
    for row in rows:
        t = torch.tensor([row["target_distribution"][c] for c in row["candidates"]])
        m = row_metrics(preds[row["id"]], t)
        m["id"] = row["id"]
        items.append(m)
        groups[(row["mechanism"], row["difficulty"])].append(m)
    report = {
        "tag": "nanojev_on_xiaojev_test",
        "overall": summarize(items),
        "by_mechanism_difficulty": {f"{m}/{d}": summarize(v) for (m, d), v in sorted(groups.items())},
        "paired_consistency": paired_consistency(rows, preds),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".preds.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps({"id": row["id"],
                                "probs": [round(float(x), 6) for x in preds[row["id"]]]}) + "\n")
    print(json.dumps(report["overall"], indent=1))
    print("paired:", report["paired_consistency"])
    print("->", out)


if __name__ == "__main__":
    main()
