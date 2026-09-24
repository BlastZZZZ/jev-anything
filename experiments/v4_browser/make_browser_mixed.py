"""Programmatic mixed-control pages using the real browser request serializer."""

import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("XIAOJEV_BROWSER_RUN", str(ROOT / "results/v4_browser")))
sys.path.insert(
    0, os.environ.get("XIAOJEV_JEV_HOME", str(ROOT.parent / "jev-ultrafast"))
)
from jev_ultrafast.local_model import question_row
from jev_ultrafast.model import decision_request


def generate(seed, n):
    rng = random.Random(seed)
    cities = [
        "Bergen",
        "Kyoto",
        "Dublin",
        "Prague",
        "Rome",
        "Berlin",
        "Seoul",
        "Oslo",
        "Porto",
        "Tallinn",
    ]
    names = [
        "Aster House",
        "Cedar Court",
        "Moss Row",
        "Villa Nova",
        "Atlas House",
        "River Loft",
        "Harbor Light",
    ]
    rows = []
    for sid in range(n):
        city = rng.choice(cities)
        other_city = rng.choice([x for x in cities if x != city])
        name = rng.choice(names)
        other_name = rng.choice([x for x in names if x != name])
        category = rng.choice(["Design", "Nature", "Coastal", "Urban"])
        categories = ["Design", "Nature", "Coastal", "Urban"]
        free = rng.choice([True, False])
        require_free = free or rng.random() < 0.4
        scenario = rng.choice(
            [
                "empty",
                "wrong",
                "typed",
                "category",
                "checkbox",
                "results",
                "done",
                "detail_wrong",
                "autocomplete",
                "loading",
                "empty",
                "results",
            ]
        )
        label = rng.choice(["Destination", "City", "Location", "Search destination"])
        submit = rng.choice(["Find stays", "Search", "Show results", "Search listings"])
        brand = rng.choice(["Forma", "Haven", "Terra", "Roam", "Daybreak"])
        goal = f"Use the {label.lower()} search and filters to find {category} stays in {city}"
        if require_free:
            goal += (
                " with Free cancellation"
                if free
                else " with Free cancellation turned off"
            )
        goal += f", then open {name}."
        value = (
            "" if scenario == "empty" else other_city if scenario == "wrong" else city
        )
        current_category = (
            "All stays"
            if scenario in ("empty", "wrong", "typed", "category", "autocomplete")
            else category
        )
        checked = (
            not free
            if scenario
            in ("empty", "wrong", "typed", "category", "checkbox", "autocomplete")
            else free
        )
        searched = scenario not in ("empty", "wrong", "typed", "autocomplete")
        actions = []
        counter = 0

        def element(label, role, kinds=("click",), **props):
            nonlocal counter
            counter += 1
            node = counter
            ids = {}
            for kind in kinds:
                action = dict(
                    id=f"e{node}_{kind}",
                    node=node,
                    kind=kind,
                    label=label,
                    role=role,
                    value="",
                    **{},
                )
                action.update(props)
                actions.append(action)
                ids[kind] = action["id"]
            return ids

        element(brand.lower() + ".", "link")
        element(rng.choice(["Find a stay", "Browse stays", "Explore"]), "link")
        element(rng.choice(["Reading room", "Travel journal", "About us"]), "link")
        text = [
            brand.lower() + ".",
            "Find a stay",
            "Reading room",
            "PLACES WITH CHARACTER",
            "Somewhere you can exhale.",
            "Small, considered places. A little closer to what matters.",
            "CURATED, NOT CROWDED",
        ]
        url = f"https://{brand.lower()}.example/stays"
        history = []
        correct_target = None
        if scenario in ("done", "detail_wrong"):
            correct_target = element("← All stays", "button")["click"]
            detail_category = (
                category
                if scenario == "done"
                else rng.choice([x for x in categories if x != category])
            )
            text += [
                city.upper() + " · " + category.upper(),
                name,
                "A quiet courtyard and light all afternoon.",
                "Free cancellation included" if free else "Non-refundable stay",
                f"Your filters: {detail_category} · Free cancellation {'enabled' if checked else 'off'} · Destination {city}",
            ]
            title = name + " · " + brand
            url += "#" + name.lower().replace(" ", "-")
            operation = "DONE" if scenario == "done" else "CLICK"
        else:
            role = (
                "combobox"
                if scenario == "autocomplete"
                else rng.choice(["searchbox", "textbox", "combobox"])
            )
            field = element(label, role, ("fill", "click"), value=value)
            go = element(submit, "button")["click"]
            counter += 1
            select_node = counter
            select_ids = {}
            for opt in categories:
                if opt == current_category:
                    continue
                aid = f"e{select_node}_select_{opt}"
                actions.append(
                    dict(
                        id=aid,
                        node=select_node,
                        kind="select",
                        label="Stay category → " + opt,
                        role="combobox",
                        value=opt,
                        current_value=current_category,
                    )
                )
                select_ids[opt] = aid
            check = element(
                "Free cancellation",
                "checkbox",
                checked=str(checked).lower(),
                value="on",
            )["click"]
            text += [submit, "Category", "Free cancellation"]
            if searched:
                text.append(
                    "1 place in " + city
                    if scenario == "results"
                    else "3 places in " + city
                )
            else:
                text.append("3 places")
            if scenario == "loading":
                text.append(
                    "Loading results. Please wait; the requested results are not available yet."
                )
                target = None
            else:
                order = [
                    (city, category, name),
                    (other_city, category, other_name),
                    (
                        city,
                        rng.choice([c for c in categories if c != category]),
                        "Forest Cabin",
                    ),
                ]
                rng.shuffle(order)
                for c, cat, nm in order:
                    text += [
                        c.upper() + " · " + cat.upper(),
                        nm,
                        "Free cancellation · 2 guests",
                        "€145 / night",
                    ]
                    aid = element("View " + nm, "button")["click"]
                    if nm == name:
                        target = aid
            if scenario == "autocomplete":
                suggestions = [
                    city + ", Country",
                    other_city + ", Country",
                    rng.choice(cities) + " Central",
                ]
                rng.shuffle(suggestions)
                for suggestion in suggestions:
                    aid = element(suggestion, "option")["click"]
                    if suggestion == city + ", Country":
                        suggestion_id = aid
                text += ["Suggestions"] + suggestions
                operation, correct_target = "CLICK", suggestion_id
            elif scenario in ("empty", "wrong"):
                operation, correct_target = "TYPE_TEXT", field["fill"]
                if rng.random() < 0.35:
                    history = [
                        dict(
                            action=submit,
                            kind="click",
                            text=None,
                            page_changed=True,
                            progressed=False,
                        )
                    ] * rng.randint(1, 3)
            elif scenario == "typed":
                operation, correct_target = "CLICK", go
                history = [
                    dict(
                        action=label,
                        kind="fill",
                        text=city,
                        page_changed=True,
                        progressed=True,
                    )
                ]
            elif scenario == "category":
                operation, correct_target = "SELECT", select_ids[category]
                history = [
                    dict(
                        action=submit,
                        kind="click",
                        text=None,
                        page_changed=True,
                        progressed=True,
                    )
                ]
            elif scenario == "checkbox" and require_free:
                operation, correct_target = "CLICK", check
            elif scenario == "loading":
                operation = "WAIT"
            else:
                operation, correct_target = "CLICK", target
                history = [
                    dict(
                        action=submit,
                        kind="click",
                        text=None,
                        page_changed=True,
                        progressed=True,
                    )
                ]
            title = brand + " · Find a place to slow down"
        if rng.random() < 0.5:
            actions.append(dict(id="scroll_down", kind="scroll", label="Scroll down"))
        actions.append(dict(id="wait", kind="wait", label="Wait"))
        page = dict(url=url, title=title, text="\n".join(text), actions=actions)
        body, targets, _ = decision_request(page, goal, history)
        h = (
            int(hashlib.sha256(f"mixed:{seed}:{sid}".encode()).hexdigest()[:8], 16)
            % 100
        )
        split = "train" if h < 85 else "dev"
        labels = [("operation", operation)]
        if operation in targets:
            index = next(
                i for i, a in targets[operation].items() if a["id"] == correct_target
            )
            labels.append((operation.lower() + "_target", index))
        for question_name, gold in labels:
            question = body["questions"][question_name]
            row = question_row(body["state"], question)
            keys = list(question["criteria"])
            row.update(
                id=f"browser_mixed_{sid}_{question_name}",
                family="browser",
                mechanism="browser_mixed_" + scenario,
                difficulty="compositional",
                split=split,
                proposition=None,
                paired_id=None,
                target_distribution={
                    c: float(k == gold) for k, c in zip(keys, row["candidates"])
                },
                meta=dict(
                    scenario_id=sid,
                    phase=scenario,
                    question_kind=question_name,
                    correct_key=gold,
                    goal=goal,
                    source="programmatic mixed-control pages",
                ),
            )
            assert len(set(row["candidates"])) == len(row["candidates"])
            rows.append(row)
    return rows


if __name__ == "__main__":
    rows = generate(20260924, 9000)
    with (OUT / "browser_mixed.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("rows", len(rows), "splits", Counter(r["split"] for r in rows))
    print("phases", Counter(r["meta"]["phase"] for r in rows))
