"""Convert NanoJev-Data unified/soft dataset into the xiaojev train.py JSONL format.

Supervision: maze/snake rows carry Jev teacher native_probs; shooting rows
carry the visual expert's policy_probs. Both are full soft distributions over
the action candidates; renormalized to sum 1. Splits are kept as-is.

Input: the NanoJev-Data dataset (https://huggingface.co/datasets/C-Tianyu/NanoJev-Data),
directory containing unified/soft/{train,dev,calibration,test,ood}.jsonl.
Pass --src, or set XIAOJEV_NANOJEV_DATA to the dataset root (the directory
that contains unified/), or leave both unset to download a snapshot with
huggingface_hub.
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "games_v1.jsonl"

SPLITS = ["train", "dev", "calibration", "test", "ood"]


def resolve_src(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("XIAOJEV_NANOJEV_DATA")
    if env:
        return Path(env) / "unified" / "soft"
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("C-Tianyu/NanoJev-Data", repo_type="dataset")) / "unified" / "soft"


def mechanism(meta):
    if meta["task"] == "shooting":
        return "game_doom_" + meta["spec"].get("scenario", "unknown")
    return "game_" + meta["task"]


def convert(src, out):
    counts = Counter()
    rows_out = []
    for split in SPLITS:
        with open(src / f"{split}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                q = d["questions"]["action"]
                crit = q["criteria"]
                ids = list(crit)
                texts = [crit[k] for k in ids]
                if len(set(texts)) != len(texts):
                    raise ValueError(f"duplicate candidate text in {d['id']}")
                if "teacher" in d:
                    probs = d["teacher"]["native_probs"]["action"]
                    source = "jev_teacher"
                else:
                    probs = d["expert"]["policy_probs"]
                    source = "visual_expert"
                if set(probs) != set(ids):
                    raise ValueError(f"candidate/prob mismatch in {d['id']}")
                total = sum(probs.values())
                if total <= 0:
                    raise ValueError(f"zero prob sum in {d['id']}")
                target = {crit[k]: probs[k] / total for k in ids}
                meta = d["metadata"]
                rows_out.append({
                    "id": "games_" + d["id"],
                    "family": "game_decision",
                    "mechanism": mechanism(meta),
                    "primitive": "choice",
                    "state": d["state"],
                    "instruction": q["instructions"],
                    "proposition": None,
                    "candidates": texts,
                    "target_distribution": target,
                    "difficulty": "simple",
                    "split": split,
                    "paired_id": None,
                    "meta": {
                        "task": meta["task"],
                        "scenario": meta["spec"].get("scenario"),
                        "episode_id": meta["episode_id"],
                        "decision_index": meta["decision_index"],
                        "source_group_id": meta["source_group_id"],
                        "remaining_steps": meta["remaining_steps"],
                        "supervision": source,
                    },
                })
                counts[(split, mechanism(meta))] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out, len(rows_out))
    for k in sorted(counts):
        print(" ", k, counts[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=None,
                    help="NanoJev-Data unified/soft directory "
                         "(default: $XIAOJEV_NANOJEV_DATA/unified/soft, "
                         "else a huggingface_hub snapshot download)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    convert(resolve_src(args.src), Path(args.out))


if __name__ == "__main__":
    main()
