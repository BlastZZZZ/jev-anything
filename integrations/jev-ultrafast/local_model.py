"""Local xiaojev decision backend: same response shape as the TypeSafe API.

Scores each choice question's candidates with the xiaojev scoring head
(Qwen3-0.6B + EOS readout, checkpoint from XIAOJEV_CKPT) and returns
{"answers": {qid: {"choice", "probabilities", "confidence"}}, "model", "usage"}.
The model only ever returns an index over the offered candidates; it never
emits selectors, code, or text.
"""

import json
import os
import re
import sys
import threading
from pathlib import Path

_ROOT_CANDIDATES = (
    Path(__file__).resolve().parents[2],
    Path(__file__).resolve().parents[2] / "jev-anything",
    Path(__file__).resolve().parents[2] / "xiaojev",
)
_DEFAULT_HOME = next(
    (p for p in _ROOT_CANDIDATES if (p / "training/train.py").is_file()),
    _ROOT_CANDIDATES[1],
)
_HOME = os.environ.get("XIAOJEV_HOME", str(_DEFAULT_HOME))
_STATE_CHAR_CAP = 6000
_MICROBATCH_TOKENS = 16384

_lock = threading.Lock()
_engine = None


def _serialize(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "element" in value:
        match = re.fullmatch(r"\[([^]]+)\]\s*(.*)", value["element"])
        label = f'[{match[1]}] "{match[2]}"' if match else value["element"]
        extra = []
        for key in ("role", "current_value", "checked", "selected", "expanded"):
            if key in value and value[key] is not None:
                extra.append(
                    f"{key}={value[key]!r}"
                    if key == "current_value"
                    else f"{key}={str(value[key]).lower()}"
                )
        return label + (f" ({'; '.join(extra)})" if extra else "")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _state_text(state):
    parts = []
    page = state.get("page") or {}
    if page:
        parts.append(f"PAGE: {page.get('title', '')} — {page.get('url', '')}")
        if page.get("text"):
            parts.append(str(page["text"])[:2000])
    elements = state.get("elements") or []
    if elements:
        lines = []
        for e in elements:
            bits = [
                f"role={e['role']}" if e.get("role") else None,
                f"operations={','.join(e['operations'])}"
                if e.get("operations")
                else None,
                f"value={e['value']!r}" if "value" in e else None,
            ]
            bits = [b for b in bits if b]
            for key in ("checked", "selected", "expanded"):
                if key in e:
                    bits.append(f"{key}={str(e[key]).lower()}")
            if e.get("options"):
                bits.append("options=" + json.dumps(e["options"], ensure_ascii=False))
            lines.append(f'[{e["index"]}] "{e.get("label", "")}" ({"; ".join(bits)})')
        parts.append("ELEMENTS:\n" + "\n".join(lines))
    recent = state.get("recent_actions") or []
    if recent:
        lines = [json.dumps(h, ensure_ascii=False) for h in recent]
        parts.append("RECENT ACTIONS:\n" + "\n".join(lines))
    text = "\n\n".join(parts)
    if len(text) > _STATE_CHAR_CAP:
        text = text[:_STATE_CHAR_CAP] + "\n... (truncated)"
    return text


def _instruction_text(question):
    instr = question.get("instructions")
    if isinstance(instr, dict):
        order = ["goal", "operation", "rules"]
        lines = []
        for k in order:
            if k in instr:
                value = instr[k]
                text = (
                    "\n\n".join(map(_serialize, value))
                    if isinstance(value, list)
                    else _serialize(value)
                )
                lines.append(f"{k.upper()}:\n{text}")
        for k, v in instr.items():
            if k not in order:
                lines.append(f"{k.upper()}:\n{_serialize(v)}")
        return "\n".join(lines)
    return _serialize(instr)


def question_row(state, question):
    return {
        "primitive": "choice",
        "state": _state_text(state),
        "instruction": _instruction_text(question),
        "candidates": [_serialize(c) for c in question["criteria"].values()],
    }


def _load():
    global _engine
    with _lock:
        if _engine is None:
            sys.path.insert(0, os.path.join(_HOME, "training"))
            import torch  # noqa: F401
            from train import MODEL_PATH, StudentModel, load_ckpt
            from transformers import AutoTokenizer

            ckpt = os.environ.get(
                "XIAOJEV_CKPT", os.path.join(_HOME, "ckpt/v4_browser")
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = StudentModel().to("cuda")
            load_ckpt(model, ckpt)
            model.eval()
            _engine = (tokenizer, model, ckpt)
    return _engine


def answer(body):
    sys.path.insert(0, os.path.join(_HOME, "training"))
    import torch
    from train import collate, encode_row

    tokenizer, model, ckpt = _load()
    paths, spans = [], []
    for qid, question in body["questions"].items():
        if question.get("type", "choice") != "choice":
            raise ValueError(f"Local backend only scores choice questions; got {qid!r}")
        criteria = question["criteria"]
        keys = list(criteria)
        row = question_row(body["state"], question)
        _, _, row_paths = encode_row(tokenizer, row)
        spans.append((qid, keys, len(paths), len(row_paths)))
        paths.extend(row_paths)
    order = sorted(range(len(paths)), key=lambda i: -len(paths[i]))
    chunks, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, len(paths[i]))
        if cur and new_max * (len(cur) + 1) > _MICROBATCH_TOKENS:
            chunks.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, len(paths[i]))
    if cur:
        chunks.append(cur)
    scores = torch.empty(len(paths), dtype=torch.float32)
    for chunk in chunks:
        tokens, mask, lengths = collate(
            [paths[i] for i in chunk], tokenizer.pad_token_id, "cuda"
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            part = model(tokens, mask, lengths)
        scores[chunk] = part.float().cpu()
    answers = {}
    for qid, keys, offset, k in spans:
        probs = torch.softmax(scores[offset : offset + k].double(), dim=-1)
        probs = (probs / probs.sum()).tolist()
        dist = dict(zip(keys, probs))
        choice = max(range(k), key=lambda i: (probs[i], -i))
        top = probs[choice]
        confidence = 1.0 if k == 1 else (top - 1 / k) / (1 - 1 / k)
        answers[qid] = {
            "choice": keys[choice],
            "probabilities": dist,
            "confidence": confidence,
        }
    return {
        "answers": answers,
        "model": f"xiaojev-local:{os.path.basename(ckpt.rstrip('/'))}",
        "usage": {
            "questions": len(spans),
            "candidate_paths": len(paths),
            "forward_passes": len(chunks),
            "autoregressive_decode_steps": 0,
            "prompt_tokens_estimate": sum(map(len, paths)),
        },
    }
