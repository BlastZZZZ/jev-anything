"""System One primitives on a local vLLM server via candidate scoring.

Each candidate label is appended to the chat-templated prompt and scored with
/v1/completions prompt_logprobs (exact, works for any K, immune to the
top_logprobs=20 cap and to guided decoding not masking returned logprobs).
The per-candidate logprobs are softmaxed into a probability distribution.

Prefix caching makes the K scoring passes cheap: the long state prefix is
computed once. Every request/response is appended to a JSONL journal.

Configuration (environment variables):
  XIAOJEV_VLLM_URL    OpenAI-compatible base URL (default http://127.0.0.1:8020/v1)
  XIAOJEV_VLLM_MODEL  served model name (default qwen3.8-27b)
  XIAOJEV_MODEL_PATH  tokenizer path or HF id (default cyankiwi/Qwen3.8-27B-AWQ-INT4)
  XIAOJEV_JOURNAL     journal JSONL path (default <repo>/results/journal.jsonl)

We served Qwen3.8-27B-AWQ-INT4 with vLLM for the paper's probes; any
OpenAI-compatible server that supports prompt_logprobs works.
"""

import hashlib
import json
import math
import os
import string
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_URL = os.environ.get("XIAOJEV_VLLM_URL", "http://127.0.0.1:8020/v1")
MODEL = os.environ.get("XIAOJEV_VLLM_MODEL", "qwen3.8-27b")
MODEL_PATH = os.environ.get("XIAOJEV_MODEL_PATH", "cyankiwi/Qwen3.8-27B-AWQ-INT4")
JOURNAL = os.environ.get("XIAOJEV_JOURNAL", str(REPO_ROOT / "results" / "journal.jsonl"))

# Single-token labels: 0-9 then A-Z (verified against /tokenize).
LABELS = list(string.digits + string.ascii_uppercase)

_tokenizer = None


def _tok():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return _tokenizer


class DistributionError(Exception):
    pass


def _chat_text(messages):
    return _tok().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _score_label(client, prefix, label):
    """Logprob of `label` as the next token(s) after prefix."""
    body = {
        "model": MODEL,
        "prompt": prefix + label,
        "max_tokens": 1,
        "prompt_logprobs": 1,
    }
    resp = client.post(f"{BASE_URL}/completions", json=body, timeout=60)
    data = resp.json()
    if resp.status_code != 200:
        raise DistributionError(f"HTTP {resp.status_code}: {data}")
    positions = data["choices"][0]["prompt_logprobs"]
    n_label_tokens = len(_tok().encode(label))
    # Sum logprobs over the label's tokens (last n positions).
    total = 0.0
    for pos in positions[-n_label_tokens:]:
        entry = next(iter(pos.values()))
        total += entry["logprob"]
    return total


def _decide(messages, labels, tag):
    if len(labels) > len(LABELS):
        raise ValueError(f"too many candidates: {len(labels)}")
    prefix = _chat_text(messages)
    started = time.perf_counter()
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=8) as pool:
            logprobs = list(
                pool.map(lambda l: _score_label(client, prefix, l), labels)
            )
    m = max(logprobs)
    exps = {l: math.exp(lp - m) for l, lp in zip(labels, logprobs)}
    total = sum(exps.values())
    dist = {l: v / total for l, v in exps.items()}
    record = {
        "tag": tag,
        "request_sha256": hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "labels": labels,
        "candidate_logprobs": dict(zip(labels, logprobs)),
        "distribution": dist,
        "chosen": labels[logprobs.index(m)],
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
    Path(JOURNAL).parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dist, record


def choice(state, instruction, candidates, tag="choice"):
    """candidates: list of description strings. Returns ({label: prob}, record)."""
    labels = LABELS[: len(candidates)]
    table = "\n".join(f"[{l}] {c}" for l, c in zip(labels, candidates))
    prompt = (
        "You are a decision system. Read the state, then answer the question by "
        "choosing exactly one option. Judge the true probability of each option "
        "from the state; do not default to uniform and do not be overconfident.\n\n"
        f"STATE:\n{state}\n\nQUESTION:\n{instruction}\n\nOPTIONS:\n{table}\n\n"
        "Answer with only the option label."
    )
    return _decide([{"role": "user", "content": prompt}], labels, tag)


def noul(state, proposition, tag="noul"):
    """Returns (p_yes, record)."""
    prompt = (
        "You are a decision system. Read the state, then judge whether the "
        "statement is true. Answer with only yes or no. Judge the true "
        "probability; do not be overconfident.\n\n"
        f"STATE:\n{state}\n\nSTATEMENT:\n{proposition}\n\n"
        "Answer with only yes or no."
    )
    dist, record = _decide(
        [{"role": "user", "content": prompt}], ["yes", "no"], tag
    )
    return dist["yes"], record


def score(state, instruction, levels, tag="score"):
    """levels: ordered level descriptions. Returns (expectation, dist, record)."""
    dist, record = choice(state, instruction, levels, tag=tag)
    expectation = sum(i * dist[l] for i, l in enumerate(LABELS[: len(levels)]))
    return expectation, dist, record
