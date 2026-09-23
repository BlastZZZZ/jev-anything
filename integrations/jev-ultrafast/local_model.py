"""Local xiaojev decision backend: same response shape as the TypeSafe API.

Scores each choice question's candidates with the xiaojev scoring head
(Qwen3-0.6B + EOS readout, checkpoint from XIAOJEV_CKPT) and returns
{"answers": {qid: {"choice", "probabilities", "confidence"}}, "model", "usage"}.
The model only ever returns an index over the offered candidates; it never
emits selectors, code, or text.

Configuration (environment variables):
  XIAOJEV_HOME  xiaojev repo root (default: auto-detected relative to this file)
  XIAOJEV_CKPT  checkpoint dir (default: <XIAOJEV_HOME>/ckpt/v3)
"""

import json
import os
import sys
import threading
from pathlib import Path

_HOME = os.environ.get(
    "XIAOJEV_HOME", str(Path(__file__).resolve().parents[2])
)
_STATE_CHAR_CAP = 6000
_MICROBATCH_TOKENS = 16384

_lock = threading.Lock()
_engine = None


def _serialize(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "element" in value:
        extra = "; ".join(f"{k}={v}" for k, v in value.items()
                          if k != "element" and v not in (None, "", False))
        return value["element"] + (f" ({extra})" if extra else "")
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
            bits = [f"role={e['role']}" if e.get("role") else None,
                    f"operations={','.join(e['operations'])}" if e.get("operations") else None,
                    f"value={e['value']!r}" if e.get("value") else None,
                    f"options={len(e['options'])}" if e.get("options") else None]
            bits = [b for b in bits if b]
            lines.append(f"[{e['index']}] \"{e.get('label', '')}\" ({'; '.join(bits)})")
        parts.append("ELEMENTS:\n" + "\n".join(lines))
    recent = state.get("recent_actions") or []
    if recent:
        lines = [f"- {h.get('kind') or ''} \"{h.get('action') or ''}\""
                 + (" (page changed)" if h.get("page_changed") else "") for h in recent]
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
                lines.append(f"{k.upper()}:\n{_serialize(instr[k])}")
        for k, v in instr.items():
            if k not in order:
                lines.append(f"{k.upper()}:\n{_serialize(v)}")
        return "\n".join(lines)
    return _serialize(instr)


def _load():
    global _engine
    with _lock:
        if _engine is None:
            sys.path.insert(0, os.path.join(_HOME, "training"))
            import torch  # noqa: F401
            from transformers import AutoTokenizer
            from train import MODEL_PATH, StudentModel, load_ckpt
            ckpt = os.environ.get("XIAOJEV_CKPT", os.path.join(_HOME, "ckpt/v3"))
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
    state_text = _state_text(body["state"])
    paths, spans = [], []
    for qid, question in body["questions"].items():
        if question.get("type", "choice") != "choice":
            raise ValueError(f"Local backend only scores choice questions; got {qid!r}")
        criteria = question["criteria"]
        keys = list(criteria)
        row = {
            "primitive": "choice",
            "state": state_text,
            "instruction": _instruction_text(question),
            "candidates": [_serialize(criteria[k]) for k in keys],
        }
        _, _, row_paths = encode_row(tokenizer, row)
        spans.append((qid, keys, len(paths), len(row_paths)))
        paths.extend(row_paths)
    tokens, mask, lengths = collate(paths, tokenizer.pad_token_id, "cuda")
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
        idx = torch.tensor(chunk, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            part = model(tokens[idx], mask[idx], lengths[idx])
        scores[idx.cpu()] = part.float().cpu()
    answers = {}
    for qid, keys, offset, k in spans:
        probs = torch.softmax(scores[offset : offset + k].double(), dim=-1)
        probs = (probs / probs.sum()).tolist()
        dist = dict(zip(keys, probs))
        choice = max(range(k), key=lambda i: (probs[i], -i))
        top = probs[choice]
        confidence = 1.0 if k == 1 else (top - 1 / k) / (1 - 1 / k)
        answers[qid] = {"choice": keys[choice], "probabilities": dist, "confidence": confidence}
    return {
        "answers": answers,
        "model": f"xiaojev-local:{os.path.basename(ckpt.rstrip('/'))}",
        "usage": {
            "questions": len(spans),
            "candidate_paths": len(paths),
            "forward_passes": len(chunks),
            "autoregressive_decode_steps": 0,
            "prompt_tokens_estimate": int(lengths.sum().item()),
        },
    }
