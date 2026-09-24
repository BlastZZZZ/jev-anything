"""Browser-interaction decision data: programmatic state-machine truth, no LLM.

Each scenario renders a synthetic web page state in jev-ultrafast's observation
shape (indexed elements + operations + recent actions) plus a goal, and emits
two question records (operation choice + target choice) whose target
distributions come from interaction rules. Splits are by scenario id hash.
"""

import hashlib
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "browser_v1.jsonl"
SEED = 20260923

OP_LABELS = {
    "CLICK": (
        "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "点击元素、按钮、菜单项、自动补全建议或日期。",
    ),
    "TYPE_TEXT": (
        "Enter or replace text in an editable field. A text model will supply the value.",
        "在可输入字段中输入或替换文本，文本内容由文本模型生成。",
    ),
    "SELECT": ("Select an observed dropdown value.", "选择一个可见的下拉选项。"),
    "SCROLL_UP": ("Scroll the page up.", "向上滚动页面。"),
    "SCROLL_DOWN": ("Scroll the page down.", "向下滚动页面。"),
    "WAIT": ("Wait for the page to load or update.", "等待页面加载或更新。"),
    "DONE": (
        "Every requirement is visibly satisfied.",
        "所有要求已在页面上可见地满足。",
    ),
    "BLOCKED": (
        "No supported operation can progress the goal.",
        "没有任何支持的操作能推进目标。",
    ),
}

RULES = {
    "en": (
        "Fill required fields before submitting. A field that already holds the requested value "
        "must not be filled again. A typed query still needs its matching autocomplete suggestion "
        "clicked. Submit populated search fields before opening a result. Do not toggle a control "
        "already in the requested state. Do not repeat an action that made no progress. DONE "
        "requires visible evidence that every requirement is satisfied."
    ),
    "zh": (
        "先填写必填字段再提交。已包含目标值的字段不要重复填写。输入查询后还需点击匹配的自动补全建议。"
        "搜索字段填好后先提交再打开结果。不要切换已处于目标状态的控件。不要重复没有进展的动作。"
        "DONE 要求所有要求在页面上可见地满足。"
    ),
}

CITIES = [
    "Lisbon",
    "Porto",
    "Copenhagen",
    "Oslo",
    "Kyoto",
    "Bergen",
    "Tallinn",
    "Galway",
]
PLACES = [
    "Casa Flora",
    "The Glasshouse",
    "Serra Lodge",
    "Atlas House",
    "Moss Row",
    "Fjord Rooms",
    "Cedar Court",
    "The Long Table",
    "Harbor Light",
    "Villa Nova",
]
THEMES = ["stays", "flights", "books"]
CATEGORIES = ["Design", "Nature", "Coastal", "Urban"]


def split_of(sid):
    h = int(hashlib.sha256(f"browser:{sid}:{SEED}".encode()).hexdigest()[:8], 16) % 100
    return (
        "train" if h < 70 else "dev" if h < 80 else "calibration" if h < 90 else "test"
    )


class El:
    def __init__(self, index, label, role, ops, **kw):
        self.index, self.label, self.role, self.ops, self.kw = (
            index,
            label,
            role,
            ops,
            kw,
        )

    def line(self):
        bits = [f"role={self.role}", f"operations={','.join(self.ops)}"]
        for k in ("value", "checked", "options"):
            v = self.kw.get(k)
            if k == "options" and v:
                bits.append("options=" + "|".join(v))
            elif v not in (None, "", []):
                bits.append(f"{k}={v!r}" if k == "value" else f"{k}={str(v).lower()}")
        return f'[{self.index}] "{self.label}" ({"; ".join(bits)})'

    def candidate(self):
        return f'[{self.index}] "{self.label}" (role={self.role})'


class Page:
    """A scenario page: elements, available operations, recent actions."""

    def __init__(self, theme, lang, goal, title, text):
        self.theme, self.lang, self.goal = theme, lang, goal
        self.title, self.text = title, text
        self.elements = []
        self.recent = []
        self.scrollable = False

    def add(self, label, role, ops, **kw):
        self.elements.append(El(len(self.elements) + 1, label, role, ops, **kw))
        return self.elements[-1]

    def shuffle(self, rng):
        rng.shuffle(self.elements)
        for i, e in enumerate(self.elements):
            e.index = i + 1

    def by_label(self, label):
        return next(e for e in self.elements if e.label == label)

    def state_text(self):
        parts = [
            f"PAGE: {self.title}",
            self.text,
            "ELEMENTS:\n" + "\n".join(e.line() for e in self.elements),
        ]
        if self.recent:
            parts.append("RECENT ACTIONS:\n" + "\n".join(self.recent))
        return "\n\n".join(parts)

    def operation_candidates(self, extra_ops=()):
        ops = {}
        avail = {op for e in self.elements for op in e.ops}
        for op in ("CLICK", "TYPE_TEXT", "SELECT", "SCROLL_UP", "SCROLL_DOWN", "WAIT"):
            if (
                op in avail
                or op in extra_ops
                or op == "WAIT"
                or (self.scrollable and op in ("SCROLL_UP", "SCROLL_DOWN"))
            ):
                ops[op] = OP_LABELS[op][0 if self.lang == "en" else 1]
        ops["DONE"] = OP_LABELS["DONE"][0 if self.lang == "en" else 1]
        ops["BLOCKED"] = OP_LABELS["BLOCKED"][0 if self.lang == "en" else 1]
        return ops


def normalize(dist):
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def soft(primary, weight=0.75, others=()):
    dist = (
        {p: weight for p in primary}
        if isinstance(primary, (list, tuple))
        else {primary: weight}
    )
    rest = [o for o in others if o not in dist]
    if rest:
        for o in rest:
            dist[o] = (1 - weight) / len(rest)
    return dist


def emit(
    out, sid, page, mechanism, rule_case, op_dist, target_op, target_dist, target_cands
):
    instr_op = (
        (f"GOAL:\n{page.goal}\n\nRULES:\n{RULES[page.lang]}")
        if page.lang == "en"
        else f"目标:\n{page.goal}\n\n规则:\n{RULES[page.lang]}"
    )
    op_cands = page.operation_candidates()
    full_op = {k: op_dist.get(k, 0.0) for k in op_cands}
    assert abs(sum(full_op.values()) - 1.0) < 1e-6, (sid, full_op)
    op_winner = max(full_op, key=lambda k: (full_op[k], -list(op_cands).index(k)))
    meta_common = {
        "theme": page.theme,
        "lang": page.lang,
        "rule_case": rule_case,
        "correct_operations": [k for k, v in op_dist.items() if v > 0],
    }
    out.append(
        {
            "id": f"browser_{sid}_operation",
            "family": "browser",
            "mechanism": mechanism,
            "primitive": "choice",
            "state": page.state_text(),
            "instruction": instr_op,
            "proposition": None,
            "candidates": list(op_cands.values()),
            "target_distribution": {op_cands[k]: v for k, v in full_op.items()},
            "difficulty": "simple",
            "split": split_of(sid),
            "paired_id": None,
            "meta": {**meta_common, "question_kind": "operation", "scenario_id": sid},
        }
    )
    instr_t = (
        (
            f"GOAL:\n{page.goal}\n\nThe next operation is {target_op}. Choose the best target element."
            f"\n\nRULES:\n{RULES[page.lang]}"
        )
        if page.lang == "en"
        else f"目标:\n{page.goal}\n\n下一步操作是 {target_op}。选择最佳目标元素。\n\n规则:\n{RULES[page.lang]}"
    )
    full_target = {c: target_dist.get(c, 0.0) for c in target_cands}
    assert abs(sum(full_target.values()) - 1.0) < 1e-6, (sid, full_target)
    out.append(
        {
            "id": f"browser_{sid}_target",
            "family": "browser",
            "mechanism": mechanism,
            "primitive": "choice",
            "state": page.state_text(),
            "instruction": instr_t,
            "proposition": None,
            "candidates": target_cands,
            "target_distribution": full_target,
            "difficulty": "simple",
            "split": split_of(sid),
            "paired_id": None,
            "meta": {
                **meta_common,
                "question_kind": "target",
                "scenario_id": sid,
                "for_operation": target_op,
                "operation_argmax": op_winner,
                "correct_targets": [c for c, v in target_dist.items() if v > 0],
            },
        }
    )


def text_of(theme, lang, rng):
    if lang == "zh":
        return {
            "stays": "精心挑选的住宿。目的地搜索与筛选。",
            "flights": "搜索航班。选择出发地与目的地。",
            "books": "浏览书目。搜索标题或作者。",
        }[theme]
    return {
        "stays": "Small, considered places. Search a destination and set filters.",
        "flights": "Search flights. Choose origin and destination.",
        "books": "Browse the catalogue. Search by title or author.",
    }[theme]


def goal_text(theme, lang, city, place, category, free):
    if lang == "zh":
        base = {
            "stays": f"搜索{city}的{category}住宿"
            + ("，要求免费取消" if free else "")
            + f"，然后打开{place}",
            "flights": f"查询飞往{city}的航班",
            "books": f"找到《{place}》并打开详情",
        }[theme]
    else:
        base = {
            "stays": f"Find {category} stays in {city}"
            + (" with free cancellation" if free else "")
            + f", then open {place}",
            "flights": f"Find flights to {city}",
            "books": f"Find the book {place} and open it",
        }[theme]
    return base


def new_page(rng, sid):
    theme = rng.choice(THEMES)
    lang = "en" if rng.random() < 0.5 else "zh"
    city, place = rng.choice(CITIES), rng.choice(PLACES)
    category, free = rng.choice(CATEGORIES), rng.random() < 0.7
    goal = goal_text(theme, lang, city, place, category, free)
    page = Page(
        theme,
        lang,
        goal,
        f"Forma · {city}" if lang == "en" else f"Forma · {city}",
        text_of(theme, lang, rng),
    )
    return page, city, place, category, free


# --- mechanism builders -----------------------------------------------------


def gen_form(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    labels = (
        ["Destination", "Check-in", "Check-out"]
        if page.lang == "en"
        else ["目的地", "入住日期", "退房日期"]
    )
    wanted = {labels[0]: city, labels[1]: "2026-10-02", labels[2]: "2026-10-05"}
    n_fields = rng.randint(1, 3)
    fields = labels[:n_fields]
    case = rng.choice(["empty", "wrong", "all_filled"])
    els = []
    empties, wrongs = [], []
    for f in fields:
        if case == "empty":
            cur = ""
        elif case == "wrong":
            cur = rng.choice(["Nowhere", "2020-01-01", "wrong"])
        else:
            cur = wanted[f]
        if cur == "":
            empties.append(f)
        elif cur != wanted[f]:
            wrongs.append(f)
        els.append(page.add(f, "textbox", ["TYPE_TEXT", "CLICK"], value=cur))
    submit = page.add(
        "Find stays" if page.lang == "en" else "搜索住宿", "button", ["CLICK"]
    )
    page.shuffle(rng)
    if empties or wrongs:
        need = empties + wrongs
        op_dist = {"TYPE_TEXT": 1.0}
        target_els = [page.by_label(f) for f in need]
        t_dist = normalize({e.candidate(): 1.0 for e in target_els})
        rc = "field_empty" if empties else "field_wrong_value"
    else:
        op_dist = {"CLICK": 1.0}
        t_dist = {submit.candidate(): 1.0}
        rc = "all_fields_satisfied_submit"
    emit(
        ITEMS,
        sid,
        page,
        "browser_form",
        rc,
        op_dist,
        "TYPE_TEXT" if (empties or wrongs) else "CLICK",
        t_dist,
        [e.candidate() for e in els] if (empties or wrongs) else [submit.candidate()],
    )


def gen_autocomplete(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    field_label = "Destination" if page.lang == "en" else "目的地"
    case = rng.choice(["match_visible", "no_match", "not_loaded"])
    typed = city if case != "no_match" else city[:3]
    field = page.add(field_label, "combobox", ["TYPE_TEXT", "CLICK"], value=typed)
    sugg_els = []
    if case == "match_visible":
        match_label = (
            f"{city}, Portugal"
            if city in ("Lisbon", "Porto")
            else f"{city} City Center"
        )
        others = rng.sample([c for c in CITIES if c != city], 2)
        sugg_els = [page.add(match_label, "option", ["CLICK"])]
        sugg_els += [page.add(f"{o} Central", "option", ["CLICK"]) for o in others]
    elif case == "no_match":
        others = rng.sample([c for c in CITIES if c != city], 3)
        sugg_els = [page.add(f"{o} Central", "option", ["CLICK"]) for o in others]
    page.add("Find stays" if page.lang == "en" else "搜索住宿", "button", ["CLICK"])
    page.shuffle(rng)
    if case == "match_visible":
        op_dist = soft("CLICK", 0.85, ["TYPE_TEXT"])
        t_dist = {sugg_els[0].candidate(): 1.0}
        target_cands = [e.candidate() for e in sugg_els]
    elif case == "no_match":
        op_dist = soft("TYPE_TEXT", 0.8, ["CLICK"])
        t_dist = {field.candidate(): 1.0}
        target_cands = [field.candidate()]
    else:
        op_dist = soft("WAIT", 0.7, ["TYPE_TEXT"])
        t_dist = {field.candidate(): 1.0}
        target_cands = [field.candidate()]
        page.recent.append(f'TYPE_TEXT "{field_label}" (page unchanged)')
    emit(
        ITEMS,
        sid,
        page,
        "browser_autocomplete",
        {
            "match_visible": "typed_query_suggestion_visible",
            "no_match": "no_matching_suggestion",
            "not_loaded": "suggestions_not_yet_visible",
        }[case],
        op_dist,
        "CLICK" if case == "match_visible" else "TYPE_TEXT",
        t_dist,
        target_cands,
    )


def gen_submit(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    fl = "Destination" if page.lang == "en" else "目的地"
    case = rng.choice(["ready", "missing", "already_searched"])
    field = page.add(
        fl, "textbox", ["TYPE_TEXT", "CLICK"], value=city if case != "missing" else ""
    )
    submit = page.add(
        "Find stays" if page.lang == "en" else "搜索住宿", "button", ["CLICK"]
    )
    if case == "already_searched":
        page.recent.append('CLICK "Find stays" (page changed)')
        for _ in range(3):
            page.add(f"View {rng.choice(PLACES)}", "button", ["CLICK"])
    page.shuffle(rng)
    if case == "ready":
        op_dist, t_op = {"CLICK": 1.0}, "CLICK"
        t_dist = {submit.candidate(): 1.0}
        cands = [submit.candidate(), field.candidate()]
        rc = "fields_populated_submit_now"
    elif case == "missing":
        op_dist, t_op = {"TYPE_TEXT": 1.0}, "TYPE_TEXT"
        t_dist = {field.candidate(): 1.0}
        cands = [field.candidate()]
        rc = "required_field_missing_fill_first"
    else:
        op_dist, t_op = {"DONE": 0.0, "CLICK": 1.0}, "CLICK"
        view = [
            e
            for e in page.elements
            if e.role == "button" and e.label.startswith("View")
        ]
        pick = rng.choice(view)
        t_dist = {pick.candidate(): 1.0}
        cands = [e.candidate() for e in view]
        rc = "already_submitted_open_result"
    emit(ITEMS, sid, page, "browser_submit", rc, op_dist, t_op, t_dist, cands)


def gen_select(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    sel_label = "Stay category" if page.lang == "en" else "住宿类别"
    options = (
        ["All stays"] + CATEGORIES[:3]
        if page.lang == "en"
        else ["全部"] + CATEGORIES[:3]
    )
    wanted = category if category in CATEGORIES[:3] else CATEGORIES[0]
    wanted_opt = wanted
    case = rng.choice(["needs_change", "already_set", "missing_option"])
    current = "All stays" if case == "needs_change" else wanted_opt
    opts = list(options)
    if case == "missing_option":
        opts = [o for o in options if o != wanted_opt]
    sel = page.add(sel_label, "combobox", ["SELECT"], value=current, options=opts)
    page.shuffle(rng)
    if case == "needs_change":
        op_dist, t_op = {"SELECT": 1.0}, "SELECT"
        targets = [f'[{sel.index}:{i + 1}] "{o}"' for i, o in enumerate(opts)]
        t_dist = {t: float(o == wanted_opt) for t, o in zip(targets, opts)}
        cands = targets
        rc = "dropdown_needs_target_value"
    elif case == "already_set":
        op_dist, t_op = {"DONE": 1.0}, "DONE"
        t_dist = {OP_LABELS["DONE"][0 if page.lang == "en" else 1]: 1.0}
        cands = list(t_dist)
        rc = "dropdown_already_requested_state"
    else:
        op_dist, t_op = {"BLOCKED": 0.9, "SELECT": 0.1}, "BLOCKED"
        t_dist = {OP_LABELS["BLOCKED"][0 if page.lang == "en" else 1]: 1.0}
        cands = list(t_dist)
        rc = "requested_option_absent"
    emit(ITEMS, sid, page, "browser_select", rc, op_dist, t_op, t_dist, cands)


def gen_filter(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    cb_label = "Free cancellation" if page.lang == "en" else "免费取消"
    case = "already_set" if rng.random() < 0.5 else "needs_toggle"
    checked = free if case == "already_set" else not free
    cb = page.add(cb_label, "checkbox", ["CLICK"], checked=checked)
    submit = page.add(
        "Apply filters" if page.lang == "en" else "应用筛选", "button", ["CLICK"]
    )
    if case == "already_set":
        page.recent.append(f'CLICK "{cb_label}" (page changed)')
    page.shuffle(rng)
    if case == "needs_toggle":
        op_dist, t_op = {"CLICK": 1.0}, "CLICK"
        t_dist = {cb.candidate(): 1.0}
        cands = [cb.candidate(), submit.candidate()]
        rc = "checkbox_not_in_requested_state"
    else:
        op_dist, t_op = {"CLICK": 1.0}, "CLICK"
        t_dist = {submit.candidate(): 1.0}
        cands = [submit.candidate(), cb.candidate()]
        rc = "checkbox_already_requested_state_do_not_retoggle"
    emit(ITEMS, sid, page, "browser_filter", rc, op_dist, t_op, t_dist, cands)


def gen_done(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    case = rng.choice(["result_visible", "on_detail", "no_match"])
    if case == "on_detail":
        page.text += (
            f"\nYour filters: {category} · Free cancellation "
            f"{'enabled' if free else 'off'} · Destination {city}"
        )
        page.add(
            "← All stays" if page.lang == "en" else "← 返回列表", "button", ["CLICK"]
        )
        op_dist, t_op = {"DONE": 1.0}, "DONE"
        rc = "all_requirements_visible_done"
    elif case == "result_visible":
        target_card = page.add(f"View {place}", "button", ["CLICK"])
        page.add(
            f"View {rng.choice([p for p in PLACES if p != place])}", "button", ["CLICK"]
        )
        op_dist, t_op = {"CLICK": 1.0}, "CLICK"
        rc = "matching_result_open_it"
    else:
        page.text += (
            "\nNo places match these filters."
            if page.lang == "en"
            else "\n没有符合筛选的结果。"
        )
        page.add(
            "Clear filters" if page.lang == "en" else "清除筛选", "button", ["CLICK"]
        )
        op_dist, t_op = {"BLOCKED": 0.75, "CLICK": 0.25}, "BLOCKED"
        rc = "no_results_unsatisfiable"
    page.shuffle(rng)
    if case == "result_visible":
        cards = [e for e in page.elements if e.label.startswith("View")]
        t_dist = {target_card.candidate(): 1.0}
        cands = [e.candidate() for e in cards]
    else:
        t_dist = {OP_LABELS[t_op][0 if page.lang == "en" else 1]: 1.0}
        cands = list(t_dist)
    emit(ITEMS, sid, page, "browser_done", rc, op_dist, t_op, t_dist, cands)


def gen_nav(rng, sid):
    page, city, place, category, free = new_page(rng, sid)
    page.scrollable = True
    case = rng.choice(["scroll_fresh", "scroll_stuck"])
    nav = page.add("Reading room" if page.lang == "en" else "阅读角", "link", ["CLICK"])
    page.add("Find stays" if page.lang == "en" else "搜索住宿", "button", ["CLICK"])
    if case == "scroll_stuck":
        page.recent += ["SCROLL_DOWN (page unchanged)"] * 2
        op_dist = soft("CLICK", 0.8, ["SCROLL_DOWN"])
        t_op = "CLICK"
        rc = "repeated_scroll_no_progress_switch"
    else:
        op_dist = soft("SCROLL_DOWN", 0.8, ["CLICK"])
        t_op = "SCROLL_DOWN"
        rc = "content_below_fold_scroll"
    page.shuffle(rng)
    if case == "scroll_stuck":
        t_dist = {nav.candidate(): 1.0}
        cands = [nav.candidate()]
    else:
        t_dist = {OP_LABELS["SCROLL_DOWN"][0 if page.lang == "en" else 1]: 1.0}
        cands = list(t_dist)
    emit(ITEMS, sid, page, "browser_nav", rc, op_dist, t_op, t_dist, cands)


ITEMS = []
GENERATORS = [
    gen_form,
    gen_autocomplete,
    gen_submit,
    gen_select,
    gen_filter,
    gen_done,
    gen_nav,
]


def main():
    rng = random.Random(SEED)
    per = 2200
    for gen in GENERATORS:
        for i in range(per):
            gen(rng, f"{gen.__name__[4:]}-{i:05d}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for it in ITEMS:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter

    c = Counter(i["mechanism"] for i in ITEMS)
    print("wrote", OUT, len(ITEMS), "records")
    print(dict(c))
    print("scenarios:", len(ITEMS) // 2)


if __name__ == "__main__":
    main()
