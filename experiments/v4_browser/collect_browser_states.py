"""Render counterfactual training pages, observe actual DOM, and label from reset state."""

import copy
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("XIAOJEV_BROWSER_RUN", str(ROOT / "results/v4_browser")))
sys.path.insert(
    0, os.environ.get("XIAOJEV_JEV_HOME", str(ROOT.parent / "jev-ultrafast"))
)
os.environ.setdefault("BU_CDP_URL", "http://127.0.0.1:9222")
from jev_ultrafast.browser import Browser
from jev_ultrafast.local_model import question_row
from jev_ultrafast.model import decision_request


def permute_indices(body, gold_question, gold, rng):
    result = copy.deepcopy(body)
    elements = result["state"]["elements"]
    rng.shuffle(elements)
    mapping = {e["index"]: str(i + 1) for i, e in enumerate(elements)}

    def index(key):
        root, sep, suffix = key.partition(":")
        return mapping[root] + sep + suffix

    for el in elements:
        el["index"] = mapping[el["index"]]
        for opt in el.get("options", []):
            opt["index"] = index(opt["index"])
    for name, q in result["questions"].items():
        if name == "operation":
            continue
        criteria = {}
        for key, value in q["criteria"].items():
            value["element"] = re.sub(
                r"^\[[^]]+\]", f"[{index(key)}]", value["element"]
            )
            criteria[index(key)] = value
        q["criteria"] = dict(
            sorted(
                criteria.items(), key=lambda pair: tuple(map(int, pair[0].split(":")))
            )
        )
    return result, gold if gold_question == "operation" else index(gold)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026092402)
    rows = []
    raw = []
    browser = Browser(
        os.environ.get(
            "XIAOJEV_FIXTURE_URL", "http://127.0.0.1:8766/fixture.html?scenario=travel"
        )
    )
    try:
        for sid in range(900):
            city = rng.choice(
                [
                    "Bergen",
                    "Kyoto",
                    "Dublin",
                    "Prague",
                    "Rome",
                    "Seoul",
                    "Porto",
                    "Tallinn",
                ]
            )
            other_city = rng.choice(["Oslo", "Tokyo", "Vienna", "Berlin"])
            name = rng.choice(
                [
                    "Aster House",
                    "Cedar Court",
                    "Moss Row",
                    "Villa Nova",
                    "River Loft",
                    "Harbor Light",
                ]
            )
            other_name = rng.choice(["Pine Terrace", "Sunset Rooms", "City Court"])
            cat = rng.choice(["Design", "Nature", "Coastal"])
            free = rng.random() < 0.75
            phase = rng.choice(
                [
                    "empty",
                    "wrong",
                    "typed",
                    "category",
                    "checkbox",
                    "results",
                    "done",
                    "detail_wrong",
                ]
            )
            value = "" if phase == "empty" else other_city if phase == "wrong" else city
            current_cat = (
                rng.choice(
                    ["all"] + [c for c in ["Design", "Nature", "Coastal"] if c != cat]
                )
                if phase in ("empty", "wrong", "typed", "category")
                else cat
            )
            checked = (
                not free
                if phase in ("empty", "wrong", "typed", "category", "checkbox")
                else free
            )
            searched = phase not in ("empty", "wrong", "typed")
            if phase == "detail_wrong":
                current_cat = "all"
            items = [
                dict(
                    name=name,
                    city=city,
                    category=cat,
                    free=free,
                    price=145,
                    description="A quiet courtyard and warm afternoon light.",
                ),
                dict(
                    name=other_name,
                    city=other_city,
                    category="Design",
                    free=True,
                    price=210,
                    description="An airy room in the city.",
                ),
                dict(
                    name="Forest Cabin",
                    city=city,
                    category="Nature",
                    free=False,
                    price=120,
                    description="A hillside retreat.",
                ),
            ]
            payload = dict(
                items=items,
                query=city if searched else "",
                category=current_cat,
                free=checked,
                searched=searched,
                value=value,
                detail=phase in ("done", "detail_wrong"),
            )
            browser.evaluate(
                """(() => {
                const p = PAYLOAD;
                places.splice(0, places.length, ...p.items);
                query=p.query; category=p.category; free=p.free; searched=p.searched;
                location.hash=''; travel();
                document.getElementById('destination').value=p.value;
                if (p.detail) detail(places[0]);
                return true;
            })()""".replace("PAYLOAD", json.dumps(payload))
            )
            page = browser.observe(screenshot=False)
            goal = (
                f"Use the destination search and filters to find {cat} stays in {city}"
            )
            goal += (
                " with Free cancellation"
                if free
                else " with Free cancellation turned off"
            )
            goal += f", then open {name}."
            history = []
            if value == city:
                history.append(
                    dict(
                        action="Destination",
                        kind="fill",
                        text=city,
                        page_changed=True,
                        progressed=True,
                    )
                )
            if current_cat != "all" and rng.random() < 0.75:
                history.append(
                    dict(
                        action="Stay category → " + current_cat,
                        kind="select",
                        text=None,
                        page_changed=True,
                        progressed=True,
                    )
                )
            if searched:
                history.append(
                    dict(
                        action="Find stays",
                        kind="click",
                        text=None,
                        page_changed=True,
                        progressed=True,
                    )
                )
                if rng.random() < 0.3:
                    history += [
                        dict(
                            action="Find stays",
                            kind="click",
                            text=None,
                            page_changed=True,
                            progressed=False,
                        )
                    ] * rng.randint(1, 3)
            if phase in ("empty", "wrong"):
                op = "TYPE_TEXT"
                predicate = lambda a: a["kind"] == "fill"
            elif phase == "typed":
                op = "CLICK"
                predicate = lambda a: a["label"] == "Find stays"
            elif phase == "category":
                op = "SELECT"
                predicate = lambda a: a["kind"] == "select" and a["value"] == cat
            elif phase == "checkbox":
                op = "CLICK"
                predicate = lambda a: a.get("role") == "checkbox"
            elif phase == "results":
                op = "CLICK"
                predicate = lambda a: a["label"] == "View " + name
            elif phase == "detail_wrong":
                op = "CLICK"
                predicate = lambda a: "All stays" in a["label"]
            else:
                op = "DONE"
                predicate = None
            body, targets, _ = decision_request(page, goal, history)
            questions = [("operation", op)]
            if predicate:
                choices = [idx for idx, a in targets[op].items() if predicate(a)]
                assert len(choices) == 1, (phase, choices, page["actions"])
                questions.append((op.lower() + "_target", choices[0]))
            group_hash = (
                int(hashlib.sha256(f"browser-real:{sid}".encode()).hexdigest()[:8], 16)
                % 100
            )
            split = "train" if group_hash < 85 else "dev"
            raw.append(
                dict(sid=sid, phase=phase, split=split, body=body, labels=questions)
            )
            for variant in range(3):
                for qname, gold in questions:
                    b, g = (
                        permute_indices(body, qname, gold, rng)
                        if variant
                        else (body, gold)
                    )
                    q = b["questions"][qname]
                    row = question_row(b["state"], q)
                    row.update(
                        id=f"browser_dom_{sid}_{variant}_{qname}",
                        family="browser",
                        mechanism="browser_dom_" + phase,
                        split=split,
                        difficulty="compositional",
                        paired_id=None,
                        proposition=None,
                        target_distribution={
                            c: float(k == g)
                            for k, c in zip(q["criteria"], row["candidates"])
                        },
                        meta=dict(
                            scenario_id=f"dom_{sid}",
                            phase=phase,
                            question_kind=qname,
                            goal=goal,
                            source="rendered counterfactual DOM",
                            variant=variant,
                        ),
                    )
                    rows.append(row)
            if (sid + 1) % 100 == 0:
                print("observed", sid + 1, "states", flush=True)
    finally:
        browser.close()
    with (OUT / "browser_dom_raw.jsonl").open("w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    with (OUT / "browser_dom.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("rows", len(rows), "splits", Counter(r["split"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
