"""Recompute the published ranking results without a GPU, network, or corpus."""

import argparse
import json
import random
from pathlib import Path

from rag_eval.fusion import fuse_rankings
from rag_eval.metrics import agg, rank_metrics

ROOT = Path(__file__).resolve().parents[1]


def evaluate(rows, config):
    report = {"frozen_config": config, "splits": {}}
    for split in ("calibration", "dev", "test", "nontrain"):
        selected = sorted(
            (r for r in rows if split == "nontrain" or r["split"] == split),
            key=lambda r: r["id"],
        )
        baseline, fused = [], []
        for row in selected:
            gold = set(row["gold_docids"])
            ranking = fuse_rankings(row["dense_docids"], row["relevance"], **config)
            baseline.append(
                rank_metrics(row["dense_docids"], gold, (1, 4, 5, 10, 20, 50))
            )
            fused.append(rank_metrics(ranking, gold, (1, 4, 5, 10, 20, 50)))
        diffs = [f["r@5"] - b["r@5"] for b, f in zip(baseline, fused)]
        rng = random.Random(20260924)
        bootstrap = sorted(
            sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000)
        )
        report["splits"][split] = {
            "dense": agg(baseline),
            "fusion": agg(fused),
            "delta_r5": sum(diffs) / len(diffs),
            "paired_bootstrap_delta_r5_95pct": [bootstrap[125], bootstrap[4874]],
            "improved_questions": sum(d > 0 for d in diffs),
            "worsened_questions": sum(d < 0 for d in diffs),
        }
    return report


def calibrate(rows):
    calibration = [r for r in rows if r["split"] == "calibration"]
    trials = []
    for constant in (1, 5, 10, 20, 60):
        for weight in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1):
            config = {"reranker_weight": weight, "rank_constant": constant}
            score = sum(
                rank_metrics(
                    fuse_rankings(r["dense_docids"], r["relevance"], **config),
                    set(r["gold_docids"]),
                    (5,),
                )["r@5"]
                for r in calibration
            ) / len(calibration)
            trials.append({"config": config, "calibration_r5": score})
    return max(
        trials,
        key=lambda t: (
            t["calibration_r5"],
            -t["config"]["reranker_weight"],
            -t["config"]["rank_constant"],
        ),
    )["config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "results/v4_repair/retrieval_inputs.jsonl"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "results/v4_repair/fusion_config.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-calibration", action="store_true")
    args = parser.parse_args()
    with args.inputs.open() as handle:
        rows = [json.loads(line) for line in handle]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["split"] in {"calibration", "dev", "test"} for r in rows)
    config = json.loads(args.config.read_text())["config"]
    if args.verify_calibration:
        assert calibrate(rows) == config, (
            "Calibration does not reproduce the frozen config"
        )
    report = evaluate(rows, config)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
