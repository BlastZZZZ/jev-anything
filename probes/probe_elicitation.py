"""Elicitation-method comparison: is zero-shot calibration a prompting problem?

Same ground-truth cases, five elicitation methods:
  labels        - abstract label tokens, softmax over candidate logprobs (baseline)
  answer_words  - score the natural answer text directly (no label indirection)
  labels_permavg- labels, averaged over label<->candidate permutations
  verbalized    - let the model generate a probability number (0-100)
  sampling      - empirical frequencies from n samples at temperature 1
"""

import json
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import httpx
from systemone import (
    BASE_URL,
    JOURNAL,
    LABELS,
    MODEL,
    _chat_text,
    _decide,
    _score_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

import math
from concurrent.futures import ThreadPoolExecutor


def score_texts(messages, texts, tag):
    """Score arbitrary candidate answer strings (multi-token aware)."""
    prefix = _chat_text(messages)
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=8) as pool:
            lps = list(pool.map(lambda t: _score_label(client, prefix, t), texts))
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    total = sum(exps)
    dist = {t: v / total for t, v in zip(texts, exps)}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps({"tag": tag, "method": "score_texts",
                            "candidates": texts,
                            "candidate_logprobs": dict(zip(texts, lps)),
                            "distribution": dist}, ensure_ascii=False) + "\n")
    return dist


def method_answer_words(state, instruction, candidates, tag):
    prompt = (
        "You are a decision system. Read the state, then answer the question. "
        "Judge the true probability of each possible answer; do not default to "
        "uniform and do not be overconfident.\n\n"
        f"STATE:\n{state}\n\nQUESTION:\n{instruction}\n\n"
        "Answer with only the answer text, nothing else."
    )
    return score_texts([{"role": "user", "content": prompt}], candidates,
                       tag + "/answer_words")


def method_labels_permavg(state, instruction, candidates, tag):
    """Average the distribution over all cyclic label assignments."""
    k = len(candidates)
    accum = {i: 0.0 for i in range(k)}
    for shift in range(k):
        # candidate i gets label LABELS[(i+shift) % k]
        perm = [candidates[(i - shift) % k] for i in range(k)]
        table = "\n".join(f"[{LABELS[i]}] {perm[i]}" for i in range(k))
        prompt = (
            "You are a decision system. Read the state, then answer the question "
            "by choosing exactly one option. Judge the true probability of each "
            "option from the state; do not default to uniform and do not be "
            "overconfident.\n\n"
            f"STATE:\n{state}\n\nQUESTION:\n{instruction}\n\nOPTIONS:\n{table}\n\n"
            "Answer with only the option label."
        )
        dist, _ = _decide([{"role": "user", "content": prompt}],
                          LABELS[:k], f"{tag}/perm{shift}")
        for i in range(k):
            accum[(i - shift) % k] += dist[LABELS[i]]
    return {candidates[i]: accum[i] / k for i in range(k)}


def method_verbalized(state, proposition, tag):
    prompt = (
        "Read the state, then estimate the probability that the statement is "
        "true. Think about the underlying random mechanism.\n\n"
        f"STATE:\n{state}\n\nSTATEMENT:\n{proposition}\n\n"
        "Answer with only a number between 0 and 100 (the percentage)."
    )
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 6, "temperature": 0.0}
    resp = httpx.post(f"{BASE_URL}/chat/completions", json=body, timeout=60)
    text = resp.json()["choices"][0]["message"]["content"].strip()
    with open(JOURNAL, "a") as f:
        f.write(json.dumps({"tag": tag, "method": "verbalized",
                            "raw_text": text}, ensure_ascii=False) + "\n")
    digits = "".join(c for c in text if c.isdigit() or c == ".")
    return float(digits) / 100.0


def method_sampling(state, instruction, candidates, tag, n=50):
    labels = LABELS[: len(candidates)]
    table = "\n".join(f"[{l}] {c}" for l, c in zip(labels, candidates))
    prompt = (
        "You are a decision system. Read the state, then answer the question by "
        "choosing exactly one option.\n\n"
        f"STATE:\n{state}\n\nQUESTION:\n{instruction}\n\nOPTIONS:\n{table}\n\n"
        "Answer with only the option label."
    )
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 1.0, "n": n,
            "structured_outputs": {"choice": labels}}
    resp = httpx.post(f"{BASE_URL}/chat/completions", json=body, timeout=120)
    counts = {l: 0 for l in labels}
    unmatched = []
    for ch in resp.json()["choices"]:
        tok = ch["message"]["content"].strip()
        if tok in counts:
            counts[tok] += 1
        else:
            unmatched.append(tok)
    total = sum(counts.values())
    dist = {candidates[labels.index(l)]: c / total for l, c in counts.items()}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps({"tag": tag, "method": "sampling", "n": n,
                            "counts": counts,
                            "unmatched": unmatched}, ensure_ascii=False) + "\n")
    return dist


def tv_keys(p, ref_list):
    """p: {candidate: prob}; ref_list: probs aligned with candidate order."""
    return 0.5 * sum(abs(p.get(c, 0) - r) for c, r in zip(p, ref_list))


COIN = ("A fair coin was flipped in a sealed room. The result is covered and "
        "unknown to everyone.")
COIN70 = ("A biased coin that lands heads with probability 0.7 was flipped in a "
          "sealed room. The result is covered and unknown to everyone.")
LOT5 = ("A fair lottery machine contains exactly 5 identical balls, numbered 1 "
        "through 5. The machine mixes them thoroughly and draws one ball "
        "uniformly at random. The draw has been made but the result is hidden.")

# --- rerun only the missing sampling cases (others already in journal) ---
import sys as _sys
if "--rerun-sampling" in _sys.argv:
    results = []
    for name, state, ref_heads in [("coin", COIN, 0.5), ("coin70", COIN70, 0.7)]:
        d = method_sampling(state, "How did the coin land?",
                            ["Heads", "Tails"], name)
        results.append((name, "sampling_freq",
                        tv_keys(d, [ref_heads, 1 - ref_heads]), d))
    cands5 = [f"Ball number {i + 1}" for i in range(5)]
    d = method_sampling(LOT5, "Which ball was drawn?", cands5, "lottery_k5")
    results.append(("lottery_k5", "sampling_freq", tv_keys(d, [0.2] * 5), d))
    for name, method, err, detail in results:
        short = {k: round(v, 3) for k, v in detail.items()}
        print(f"{name:12s} {method:18s} err={err:.3f}  {short}")
    _sys.exit(0)

results = []

for name, state, ref_heads in [("coin", COIN, 0.5), ("coin70", COIN70, 0.7)]:
    cands = ["Heads", "Tails"]
    instr = "How did the coin land?"
    ref = [ref_heads, 1 - ref_heads]

    d = method_answer_words(state, instr, cands, name)
    results.append((name, "answer_words", tv_keys(d, ref), d))

    d = method_labels_permavg(state, instr, cands, name)
    results.append((name, "labels_permavg", tv_keys(d, ref), d))

    p = method_verbalized(state, "The coin landed heads.", name)
    results.append((name, "verbalized_p_heads", abs(p - ref_heads),
                    {"p_heads": p}))

    d = method_sampling(state, instr, cands, name)
    results.append((name, "sampling_freq", tv_keys(d, ref), d))

# K=5 lottery for the multi-candidate methods
cands5 = [f"Ball number {i + 1}" for i in range(5)]
instr5 = "Which ball was drawn?"
ref5 = [0.2] * 5
d = method_answer_words(LOT5, instr5, cands5, "lottery_k5")
results.append(("lottery_k5", "answer_words", tv_keys(d, ref5), d))
d = method_labels_permavg(LOT5, instr5, cands5, "lottery_k5")
results.append(("lottery_k5", "labels_permavg", tv_keys(d, ref5), d))
d = method_sampling(LOT5, instr5, cands5, "lottery_k5")
results.append(("lottery_k5", "sampling_freq", tv_keys(d, ref5), d))

print("\n=== SUMMARY (TV distance or |err|, lower is better) ===")
for name, method, err, detail in results:
    short = {k: round(v, 3) for k, v in detail.items()}
    print(f"{name:12s} {method:18s} err={err:.3f}  {short}")

out = REPO_ROOT / "results" / "elicitation_summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump([{"case": n, "method": m, "err": e, "detail": d}
               for n, m, e, d in results], f, ensure_ascii=False, indent=1)
print(f"-> {out}")
