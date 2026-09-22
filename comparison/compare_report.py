"""Build the three-way comparison report on NanoJev's frozen 548-case cohort.

Runs: vcdm (our rollout; internal codename for the xiaojev checkpoints),
selected (NanoJev frozen), jev (frozen API receipts), native (frozen untuned
Qwen). Verifies our trajectories with NanoJev's own verifier, recomputes
per-category success and weighted macro for all runs, paired McNemar vs Jev,
and per-case disagreement lists.

Requires:
  XIAOJEV_NANOJEV_REPO  local clone of https://github.com/TianyuCodings/NanoJev
  XIAOJEV_NANOJEV_DATA  NanoJev-Data dataset root (contains evaluation/)

Usage: compare_report.py [episodes.jsonl [out.json]]
"""
import argparse
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


ap = argparse.ArgumentParser()
ap.add_argument("episodes", nargs="?",
                default=str(REPO_ROOT / "results" / "compare_v1_frozen_episodes.jsonl"),
                help="our rollout episodes JSONL (default: %(default)s)")
ap.add_argument("out", nargs="?", default=None, help="output report JSON")
args = ap.parse_args()
VCDM = Path(args.episodes)
if args.out:
    OUT = args.out
elif args.episodes != ap.get_default("episodes"):
    OUT = str(VCDM.with_suffix("")) + "_report.json"
else:
    OUT = str(REPO_ROOT / "results" / "compare_v1_frozen.json")

sys.path.insert(0, str(nanojev_scripts()))
from summarize_sonic_supervision import WEIGHTS, SPLITS, category, verify_episode, registry  # noqa: E402
from summarize_appo_supervision import mcnemar_exact  # noqa: E402
from unified_game_pipeline import read_rows, file_digest  # noqa: E402

DATA = nanojev_data()

RUNS = {
    "vcdm": VCDM,
    "nanojev": DATA / "experiment/selected_test.jsonl",
    "jev": DATA / "jev/episodes.jsonl",
    "native_qwen": DATA / "native/native_test.jsonl",
}

cases = registry(DATA / "test_cases.jsonl")
report = {"cases_sha256": file_digest(DATA / "test_cases.jsonl"), "runs": {}}
eps_by_run = {}
for name, path in RUNS.items():
    episodes = read_rows(path)
    assert {e["case"]["id"] for e in episodes} == set(cases), name
    eps_by_run[name] = {e["case"]["id"]: e for e in episodes}
    run = {"episodes": len(episodes)}
    if name == "vcdm":
        policy = json.loads(path.with_suffix(".manifest.json").read_text())["policy"]
        run["transitions_verified"] = sum(
            verify_episode(e, cases[e["case"]["id"]], policy) for e in episodes)
        run["episode_sha256"] = file_digest(path)
    cells, succ = Counter(), Counter()
    for e in episodes:
        key = (e["case"]["split"], category(cases[e["case"]["id"]]))
        cells[key] += 1
        succ[key] += e["success"]
    run["cells"] = {f"{s}/{t}": {"successes": succ[(s, t)], "n": cells[(s, t)]}
                    for s in SPLITS for t in WEIGHTS}
    run["macro"] = {s: math.fsum(WEIGHTS[t] * succ[(s, t)] / cells[(s, t)] for t in WEIGHTS)
                    for s in SPLITS}
    report["runs"][name] = run
    print(name, "macro:", {k: round(v, 4) for k, v in run["macro"].items()},
          {k: v["successes"] for k, v in run["cells"].items()})


def paired(a, b, split, cat):
    ids = [c["id"] for c in cases.values()
           if c["split"] == split and category(c) == cat]
    wins = sorted(i for i in ids if a[i]["success"] and not b[i]["success"])
    losses = sorted(i for i in ids if not a[i]["success"] and b[i]["success"])
    return {"n": len(ids), "wins": len(wins), "losses": len(losses),
            "mcnemar_p": mcnemar_exact(len(wins), len(losses)),
            "win_case_ids": wins, "loss_case_ids": losses}


for other in ("vcdm", "nanojev", "native_qwen"):
    report["runs"][other]["paired_vs_jev"] = {
        f"{s}/{t}": paired(eps_by_run[other], eps_by_run["jev"], s, t)
        for s in SPLITS for t in WEIGHTS}
    report["runs"][other]["macro_delta_vs_jev"] = {
        s: math.fsum(WEIGHTS[t] * report["runs"][other]["paired_vs_jev"][f"{s}/{t}"]["wins"]
                     / report["runs"][other]["paired_vs_jev"][f"{s}/{t}"]["n"]
                     - WEIGHTS[t] * report["runs"][other]["paired_vs_jev"][f"{s}/{t}"]["losses"]
                     / report["runs"][other]["paired_vs_jev"][f"{s}/{t}"]["n"] for t in WEIGHTS)
        for s in SPLITS}

report["runs"]["vcdm"]["paired_vs_nanojev"] = {
    f"{s}/{t}": paired(eps_by_run["vcdm"], eps_by_run["nanojev"], s, t)
    for s in SPLITS for t in WEIGHTS}

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text(json.dumps(report, indent=2) + "\n")
print("->", OUT)
