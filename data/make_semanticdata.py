"""Build the semantic/RAG decision domain (v3 third domain) from local
hotpotqa / 2wikimultihopqa / musique with gold labels. No LLM calls.

Four task types, output compatible with datagen.py/train.py format:
  sem_evidence_choice   choice K=3-6: which passage answers the question
  sem_passage_relevance noul: does this passage help answer the question
  sem_answerability     noul: can the question be answered from the passages
  sem_sufficiency       noul: do the passages contain all needed information
Splits are by question id (70/10/10/10 train/dev/calibration/test); all items
derived from one question share its split. seed=20260922.

Input layout (--root or $XIAOJEV_RAG_DATA):
  <root>/hotpotqa/{raw/hotpotqa.json, gold.jsonl, corpus.jsonl}
  <root>/2wikimultihopqa/{raw/2wikimultihopqa.json, gold.jsonl, corpus.jsonl}
  <root>/musique/{raw/musique.json, gold.jsonl, corpus.jsonl}
i.e. the raw QA dataset JSON, a gold.jsonl with id/answer/supporting facts,
and a corpus.jsonl of {"title", "text"} passages per dataset.
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "semantic_v1.jsonl"
SEED = 20260922
SNIPPET = 300

INSTR_CHOICE = {
    "en": "Which of the following passages contains the information needed to answer the question above?",
    "zh": "以上哪条段落包含回答该问题所需的信息？",
}
PROP_RELEVANCE = {
    "en": "The passage above helps answer the question.",
    "zh": "以上段落有助于回答所提的问题。",
}
PROP_ANSWERABLE = {
    "en": "Given only the provided passages, the question can be answered.",
    "zh": "仅根据所给的段落集合，能够回答该问题。",
}
PROP_SUFFICIENT = {
    "en": "The provided passages contain all the information needed to determine the answer.",
    "zh": "所给段落包含了得出答案所需的全部信息。",
}


def rag_root(arg=None):
    root = arg or os.environ.get("XIAOJEV_RAG_DATA")
    if not root:
        raise SystemExit(
            "Pass --root or set XIAOJEV_RAG_DATA to a directory containing "
            "hotpotqa/, 2wikimultihopqa/ and musique/ (each with raw/<name>.json, "
            "gold.jsonl, corpus.jsonl)."
        )
    return Path(root)


def tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def split_of(ds, qid):
    h = int(hashlib.sha256(f"{ds}:{qid}:{SEED}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "dev" if h < 80 else "calibration" if h < 90 else "test"


def snippet(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:SNIPPET]


class Dataset:
    def __init__(self, name, root):
        base = Path(root) / name
        self.name = name
        raw_name = {"hotpotqa": "hotpotqa", "2wikimultihopqa": "2wikimultihopqa", "musique": "musique"}[name]
        self.raw = {r.get("_id") or r["id"]: r for r in json.load(open(base / "raw" / f"{raw_name}.json"))}
        self.gold = {g["id"]: g for g in map(json.loads, open(base / "gold.jsonl"))}
        self.corpus = [(c["title"], c["text"]) for c in map(json.loads, open(base / "corpus.jsonl"))]

    def question(self, qid):
        return self.raw[qid]["question"]

    def passages(self, qid):
        """All context passages: [(pid, title, text)]. pid stable within question."""
        r = self.raw[qid]
        if self.name == "musique":
            return [(p["idx"], p["title"], p["paragraph_text"]) for p in r["paragraphs"]]
        return [(i, t, " ".join(sents)) for i, (t, sents) in enumerate(r["context"])]

    def gold_pids(self, qid):
        """Ids (into passages()) of the supporting passages; None for unanswerable."""
        g = self.gold[qid]
        if self.name == "musique":
            if not g.get("answerable", True):
                return None
            return [d["paragraph_support_idx"] for d in g["question_decomposition"]]
        titles = {t for t, _ in g["supporting_facts"]}
        return [pid for pid, t, _ in self.passages(qid) if t in titles]

    def answer(self, qid):
        return self.gold[qid].get("answer") or ""

    def answerable(self, qid):
        g = self.gold[qid]
        return g.get("answerable", True) if self.name == "musique" else True


def make_item(ds, qid, task, variant, primitive, state, instruction, proposition,
              candidates, target, meta_extra):
    return {
        "id": f"sem_{ds.name}_{qid}_{task}_{variant}",
        "family": "semantic",
        "mechanism": f"sem_{task}",
        "primitive": primitive,
        "state": state,
        "instruction": instruction,
        "proposition": proposition,
        "candidates": candidates,
        "target_distribution": target,
        "difficulty": "simple",
        "split": split_of(ds.name, qid),
        "paired_id": None,
        "meta": {"dataset": ds.name, "question_id": qid, "lang": None, **meta_extra},
    }


def passage_block(title, text, lang):
    head = f"《{title}》" if lang == "zh" else f"[{title}]"
    return f"{head} {snippet(text)}"


def gen_for_question(ds, qid, rng):
    items = []
    q = ds.question(qid)
    pss = ds.passages(qid)
    gold_ids = ds.gold_pids(qid)
    answer = ds.answer(qid)
    lang = "en" if int(hashlib.sha256(qid.encode()).hexdigest()[:4], 16) % 2 == 0 else "zh"
    qhead = f"问题:{q}" if lang == "zh" else f"Question: {q}"

    if gold_ids:
        gold_set = set(gold_ids)
        distractors = [p for p in pss if p[0] not in gold_set]

        # 1. sem_evidence_choice x2 (different K / negative draws)
        for variant, k in ((0, rng.randint(3, 5)), (1, rng.randint(4, 6))):
            g_pid, g_title, g_text = pss[rng.choice(gold_ids)]
            pool = list(distractors)
            rng.shuffle(pool)
            chosen = pool[: k - 1]
            if len(chosen) < k - 1:
                extra = random_corpus_negatives(ds, q, gold_titles(ds, qid), answer, k - 1 - len(chosen), rng)
                chosen += [(-1 - i, t, x) for i, (t, x) in enumerate(extra)]
            options = [(g_pid, g_title, g_text)] + chosen
            rng.shuffle(options)
            cands = [passage_block(t, x, lang) for _, t, x in options]
            gold_idx = next(i for i, (pid, _, _) in enumerate(options) if pid == g_pid)
            items.append(make_item(ds, qid, "evidence_choice", variant, "choice", qhead,
                                   INSTR_CHOICE[lang], None, cands,
                                   {c: float(i == gold_idx) for i, c in enumerate(cands)},
                                   {"answer": answer, "gold_title": g_title, "k": len(cands),
                                    "negatives": "same-question distractors first, filtered random corpus as fill"}))

        # 2. sem_passage_relevance: one positive (gold) + one negative (filtered random)
        g_pid, g_title, g_text = pss[rng.choice(gold_ids)]
        state = f"{qhead}\n\n{passage_block(g_title, g_text, lang)}"
        items.append(make_item(ds, qid, "passage_relevance", "pos", "noul", state, None,
                               PROP_RELEVANCE[lang], ["yes", "no"], {"yes": 1.0, "no": 0.0},
                               {"answer": answer, "gold_title": g_title, "negative_filter": None}))
        neg = random_corpus_negatives(ds, q, gold_titles(ds, qid), answer, 1, rng)
        if neg:
            t, x = neg[0]
            state = f"{qhead}\n\n{passage_block(t, x, lang)}"
            items.append(make_item(ds, qid, "passage_relevance", "neg", "noul", state, None,
                                   PROP_RELEVANCE[lang], ["yes", "no"], {"yes": 0.0, "no": 1.0},
                                   {"answer": answer, "neg_title": t,
                                    "negative_filter": "no title-token overlap with question, not a gold title, answer string absent"}))

        # 3/4. answerability + sufficiency, each as a yes/no pair via gold-passage removal
        for task, prop in (("answerability", PROP_ANSWERABLE), ("sufficiency", PROP_SUFFICIENT)):
            full = "\n".join(passage_block(t, x, lang) for _, t, x in pss)
            state = f"{qhead}\n\n{full}"
            items.append(make_item(ds, qid, task, "yes", "noul", state, None, prop[lang],
                                   ["yes", "no"], {"yes": 1.0, "no": 0.0},
                                   {"answer": answer, "construction": "full context"}))
            drop = rng.choice(gold_ids)
            rest = [p for p in pss if p[0] != drop]
            drop_title = next(t for pid, t, _ in pss if pid == drop)
            partial = "\n".join(passage_block(t, x, lang) for _, t, x in rest)
            state = f"{qhead}\n\n{partial}"
            items.append(make_item(ds, qid, task, "no", "noul", state, None, prop[lang],
                                   ["yes", "no"], {"yes": 0.0, "no": 1.0},
                                   {"answer": answer, "construction": "removed gold passage",
                                    "removed_title": drop_title}))
    else:
        # musique native unanswerable: keep the full paragraph set, label no
        full = "\n".join(passage_block(t, x, lang) for _, t, x in pss)
        state = f"{qhead}\n\n{full}"
        items.append(make_item(ds, qid, "answerability", "native_no", "noul", state, None,
                               PROP_ANSWERABLE[lang], ["yes", "no"], {"yes": 0.0, "no": 1.0},
                               {"answer": answer, "construction": "musique native answerable=false"}))
    return items


def gold_titles(ds, qid):
    g = ds.gold[qid]
    if ds.name == "musique":
        ids = ds.gold_pids(qid)
        return {t for pid, t, _ in ds.passages(qid) if pid in set(ids or [])}
    return {t for t, _ in g["supporting_facts"]}


def random_corpus_negatives(ds, question, gtitles, answer, n, rng, tries=40):
    """Random corpus passages filtered to be very likely irrelevant."""
    q_toks = tokens(question)
    ans = answer.lower().strip()
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        t, x = rng.choice(ds.corpus)
        if t in gtitles:
            continue
        if tokens(t) & q_toks:
            continue
        if t.lower() in question.lower() or question.lower() in t.lower():
            continue
        if ans and len(ans) >= 3 and ans in x.lower():
            continue
        out.append((t, x))
    return out


def main():
    global SEED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="RAG datasets root (default: $XIAOJEV_RAG_DATA)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    SEED = args.seed

    rng = random.Random(SEED)
    root = rag_root(args.root)
    datasets = [Dataset(n, root) for n in ("hotpotqa", "2wikimultihopqa", "musique")]
    items = []
    for ds in datasets:
        for qid in ds.raw:
            items.extend(gen_for_question(ds, qid, rng))
    counts = Counter((i["mechanism"], i["meta"]["dataset"]) for i in items)
    splits = Counter(i["split"] for i in items)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print("wrote", out, len(items))
    for k in sorted(counts):
        print(" ", k, counts[k])
    print(" splits:", dict(splits))
    per_ds = Counter(i["meta"]["dataset"] for i in items)
    print(" per dataset:", dict(per_ds))


if __name__ == "__main__":
    main()
