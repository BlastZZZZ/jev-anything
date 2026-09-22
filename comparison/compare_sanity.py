"""Sanity check: recompute NanoJev's published 548-case comparison numbers
from the frozen public artifacts, and re-verify every controller transition
with their own verifier (NanoJev/scripts/summarize_sonic_supervision.py).

Requires:
  XIAOJEV_NANOJEV_REPO  local clone of https://github.com/TianyuCodings/NanoJev
  XIAOJEV_NANOJEV_DATA  NanoJev-Data dataset root (contains evaluation/)
"""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def nanojev_data():
    env = os.environ.get("XIAOJEV_NANOJEV_DATA")
    if not env:
        raise SystemExit(
            "Set XIAOJEV_NANOJEV_DATA to the NanoJev-Data dataset root "
            "(https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)."
        )
    return Path(env) / "evaluation"


sys.path.insert(0, str(nanojev_scripts()))
from summarize_sonic_supervision import WEIGHTS, SPLITS, category, verify_episode, registry  # noqa: E402
from unified_game_pipeline import read_rows, file_digest  # noqa: E402

DATA = nanojev_data()
OUT = REPO_ROOT / "results" / "compare_sanity.json"

PUBLISHED = {
    "selected": {"test": 0.6685, "ood": 0.4547},
    "jev": {"test": 0.6539, "ood": 0.4372},
    "native": {"test": 0.1539, "ood": 0.1219},
}
PUBLISHED_CELLS = {  # successes/n from docs/SONIC_PREDICT_POSITION_RESULTS.md
    ("selected", "test"): {"maze": (4, 10), "snake": (8, 8), "basic": (128, 128), "predict_position": (27, 128)},
    ("selected", "ood"): {"maze": (2, 10), "snake": (5, 8), "basic": (128, 128), "predict_position": (10, 128)},
    ("jev", "test"): {"maze": (7, 10), "snake": (8, 8), "basic": (56, 128), "predict_position": (11, 128)},
    ("jev", "ood"): {"maze": (3, 10), "snake": (6, 8), "basic": (59, 128), "predict_position": (8, 128)},
    ("native", "test"): {"maze": (2, 10), "snake": (0, 8), "basic": (56, 128), "predict_position": (11, 128)},
    ("native", "ood"): {"maze": (1, 10), "snake": (0, 8), "basic": (59, 128), "predict_position": (9, 128)},
}

RUNS = {
    "selected": DATA / "experiment/selected_test.jsonl",
    "jev": DATA / "jev/episodes.jsonl",
    "native": DATA / "native/native_test.jsonl",
}

cases = registry(DATA / "test_cases.jsonl")
case_sha = file_digest(DATA / "test_cases.jsonl")
print("cases sha256:", case_sha)

report = {"cases_sha256": case_sha, "runs": {}}
for name, path in RUNS.items():
    episodes = read_rows(path)
    assert len(episodes) == 548 and {e["case"]["id"] for e in episodes} == set(cases)
    policy = json.loads((path.with_suffix(".manifest.json")).read_text())["policy"]
    checked = 0
    cells = Counter()
    succ = Counter()
    for ep in episodes:
        checked += verify_episode(ep, cases[ep["case"]["id"]], policy)
        key = (ep["case"]["split"], category(cases[ep["case"]["id"]]))
        cells[key] += 1
        succ[key] += ep["success"]
    macro = {}
    for s in SPLITS:
        macro[s] = math.fsum(WEIGHTS[t] * succ[(s, t)] / cells[(s, t)] for t in WEIGHTS)
    run_rep = {
        "transitions_verified": checked,
        "macro": macro,
        "published_macro": PUBLISHED[name],
        "macro_matches_published": {
            s: abs(macro[s] - PUBLISHED[name][s]) < 5e-5 for s in SPLITS
        },
        "cells": {},
    }
    ok = True
    for s in SPLITS:
        for t in WEIGHTS:
            got = (succ[(s, t)], cells[(s, t)])
            want = PUBLISHED_CELLS[(name, s)][t]
            run_rep["cells"][f"{s}/{t}"] = {"recomputed": list(got), "published": list(want), "match": got == want}
            ok &= got == want
    run_rep["all_cells_match_published"] = ok
    report["runs"][name] = run_rep
    print(name, "macro:", {k: round(v, 4) for k, v in macro.items()}, "cells all match:", ok, "transitions:", checked)

report["passed"] = all(
    r["all_cells_match_published"] and all(r["macro_matches_published"].values())
    for r in report["runs"].values()
)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text(json.dumps(report, indent=2) + "\n")
print("passed:", report["passed"], "->", OUT)
