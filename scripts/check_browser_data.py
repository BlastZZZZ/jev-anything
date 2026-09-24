"""Tests for data/browser_v1.jsonl. Run: python3 test_browserdata.py"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "data")]
from make_browserdata import split_of, OP_LABELS  # noqa

DATA = ROOT / "data/browser_v1.jsonl"
items = [json.loads(l) for l in open(DATA)]
by_sid = defaultdict(dict)
for i in items:
    by_sid[i["meta"]["scenario_id"]][i["meta"]["question_kind"]] = i

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# pairing completeness: every scenario has exactly one operation + one target record
check(len(by_sid) * 2 == len(items), "record count != 2x scenarios")
for sid, pair in by_sid.items():
    check(set(pair) == {"operation", "target"}, f"{sid}: incomplete pair")
    op, tg = pair["operation"], pair["target"]
    check(op["split"] == tg["split"], f"{sid}: pair split mismatch")
    check(op["state"] == tg["state"], f"{sid}: pair state mismatch")
    check(op["split"] == split_of(sid), f"{sid}: split rule mismatch")
    lang = op["meta"]["lang"]
    desc2op = {OP_LABELS[k][0 if lang == "en" else 1]: k for k in OP_LABELS}
    positive_ops = {
        desc2op[c]
        for c in op["candidates"]
        if op["target_distribution"][c] > 0 and c in desc2op
    }
    check(
        tg["meta"]["for_operation"] in positive_ops,
        f"{sid}: target question for an operation with zero probability",
    )

# distribution sanity
for i in items:
    t = i["target_distribution"]
    check(set(t) == set(i["candidates"]), f"{i['id']}: target keys != candidates")
    check(abs(sum(t.values()) - 1) < 1e-9, f"{i['id']}: not normalized")
    check(all(0 <= v <= 1 for v in t.values()), f"{i['id']}: out of range")

# mechanism invariants from rule cases
for sid, pair in by_sid.items():
    op, tg = pair["operation"], pair["target"]
    rc = op["meta"]["rule_case"]
    lang2 = op["meta"]["lang"]
    d2o = {OP_LABELS[k][0 if lang2 == "en" else 1]: k for k in OP_LABELS}
    op_winner = d2o[max(op["candidates"], key=lambda c: op["target_distribution"][c])]
    t_winner = max(tg["candidates"], key=lambda c: tg["target_distribution"][c])
    st = op["state"]
    if rc == "typed_query_suggestion_visible":
        check(op_winner == "CLICK", f"{sid}: suggestion visible but not CLICK")
        check(
            "role=option" in t_winner,
            f"{sid}: target must be the matching suggestion option",
        )
        check(
            op["meta"]["lang"][:2]
            and (t_winner.split('"')[1].lower().split(",")[0] in st.lower()),
            f"{sid}: suggestion label not in state",
        )
    if rc == "suggestions_not_yet_visible":
        check(op_winner == "WAIT", f"{sid}: not-loaded case must WAIT")
    if rc in ("field_empty", "field_wrong_value"):
        check(op_winner == "TYPE_TEXT", f"{sid}: needy field must TYPE_TEXT")
    if rc == "all_fields_satisfied_submit":
        tt_desc = OP_LABELS["TYPE_TEXT"][0 if lang2 == "en" else 1]
        check(
            tt_desc not in op["candidates"] or op["target_distribution"][tt_desc] == 0,
            f"{sid}: all fields satisfied but TYPE_TEXT still probable",
        )
    if rc == "checkbox_already_requested_state_do_not_retoggle":
        check(
            "checked=true" in st or "checked=false" in st,
            f"{sid}: checkbox state missing",
        )
        check(
            "checkbox" not in t_winner, f"{sid}: must not retoggle satisfied checkbox"
        )
    if rc == "repeated_scroll_no_progress_switch":
        check(
            st.count("SCROLL_DOWN (page unchanged)") >= 2,
            f"{sid}: stuck history missing",
        )
        check(op_winner != "SCROLL_DOWN", f"{sid}: stuck scroll must switch op")
    if rc == "all_requirements_visible_done":
        check(op_winner == "DONE", f"{sid}: detail-complete must be DONE")
    if rc == "requested_option_absent":
        check(op_winner == "BLOCKED", f"{sid}: missing option must BLOCK")

# split leakage across scenarios
sid_split = {sid: next(iter(p.values()))["split"] for sid, p in by_sid.items()}
check(len(set(sid_split.values())) == 4, "missing a split")

# mechanism balance and languages
print("mechanisms:", dict(Counter(i["mechanism"] for i in items)))
print("splits:", dict(Counter(i["split"] for i in items)))
print("langs:", dict(Counter(i["meta"]["lang"] for i in items)))
print("rule cases:", dict(Counter(i["meta"]["rule_case"] for i in items)))
K = Counter(len(i["candidates"]) for i in items)
print("candidate counts:", dict(sorted(K.items())))

if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures[:15]:
        print(" -", f)
    raise SystemExit(1)
print(f"ALL CHECKS PASSED ({len(items)} records, {len(by_sid)} scenarios)")
