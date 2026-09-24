"""Run real browser tasks and independently verify filters and final pages."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("XIAOJEV_BROWSER_RUN", str(ROOT / "results/v4_browser")))
sys.path.insert(
    0, os.environ.get("XIAOJEV_JEV_HOME", str(ROOT.parent / "jev-ultrafast"))
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--tag", default="final")
    args = parser.parse_args()
    ckpt = args.checkpoint or os.environ.get(
        "XIAOJEV_CKPT", str(ROOT / "ckpt/v4_browser")
    )
    os.environ.update(
        JEV_BACKEND="local",
        XIAOJEV_HOME=str(ROOT),
        XIAOJEV_CKPT=ckpt,
        BU_CDP_URL=os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222"),
        TEXT_MODEL_BASE_URL=os.environ.get(
            "TEXT_MODEL_BASE_URL", "http://127.0.0.1:8020/v1"
        ),
        TEXT_MODEL=os.environ.get("TEXT_MODEL", "qwen3.8-27b"),
        TEXT_MODEL_API_KEY=os.environ.get("TEXT_MODEL_API_KEY", "local-dummy"),
        TEXT_MODEL_REASONING=os.environ.get("TEXT_MODEL_REASONING", "none"),
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    from jev_ultrafast import Agent

    tasks = [
        (
            "original_1",
            "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, then open Casa Flora.",
            "casa-flora",
            "Design",
            "enabled",
            "Lisbon",
        ),
        (
            "original_2",
            "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, then open Casa Flora.",
            "casa-flora",
            "Design",
            "enabled",
            "Lisbon",
        ),
        (
            "other_city",
            "Use the destination search and filters to find Design stays in Copenhagen with Free cancellation, then open The Glasshouse.",
            "the-glasshouse",
            "Design",
            "enabled",
            "Copenhagen",
        ),
        (
            "other_category",
            "Use the destination search and filters to find Nature stays in Lisbon with Free cancellation turned off, then open Serra Lodge.",
            "serra-lodge",
            "Nature",
            "off",
            "Lisbon",
        ),
    ]
    results = []
    for label, goal, slug, category, free, city in tasks:
        error = None
        state = None
        with Agent(
            os.environ.get(
                "XIAOJEV_FIXTURE_URL",
                "http://127.0.0.1:8766/fixture.html?scenario=travel",
            ),
            goal,
        ) as agent:
            try:
                for state in agent.run():
                    print(
                        label,
                        state["status"],
                        len(state["history"]),
                        state["history"][-1]["action"] if state["history"] else "",
                        flush=True,
                    )
                    if len(state["history"]) >= 15:
                        raise RuntimeError("15-action acceptance budget exceeded")
            except Exception as exc:
                error = repr(exc)
            finally:
                state = agent.snapshot()
                state["verification_text"] = agent.browser.evaluate(
                    "document.body.innerText"
                )
                (OUT / f"browser_{args.tag}_{label}.json").write_text(
                    json.dumps(state, indent=2) + "\n"
                )
        verified = (
            state["status"] == "done"
            and state["page"]["url"].endswith("#" + slug)
            and (
                f"Your filters: {category} · Free cancellation {free} · Destination {city}"
                in state["verification_text"]
            )
        )
        rec = dict(
            task=label,
            goal=goal,
            verified=verified,
            status=state["status"],
            error=error,
            actions=len(state["history"]),
            decisions=len(state["decisions"]),
            elapsed_ms=state["elapsed_ms"],
            decision_calls=len(state["decisions"]),
            text_calls=len(state["text_calls"]),
            action_sequence=[h["action"] for h in state["history"]],
        )
        results.append(rec)
        print(json.dumps(rec), flush=True)
    report = dict(
        checkpoint=ckpt,
        successes=sum(r["verified"] for r in results),
        n=len(results),
        runs=results,
    )
    (OUT / f"browser_{args.tag}_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["successes"] != report["n"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
