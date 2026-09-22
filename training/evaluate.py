"""xiaojev evaluation harness.

Two modes:
  --zero-shot   : untuned Qwen3-0.6B, candidates scored by label-token logprob
                  softmax (offline equivalent of systemone.py).
  --ckpt DIR    : trained StudentModel checkpoint (backbone/ + head.pt).

Reports accuracy, NLL, Brier, ECE(10), mean TV overall and per
mechanism x difficulty, plus paired choice-vs-noul consistency.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import (
    DATA_PATH,
    MODEL_PATH,
    StudentModel,
    collate,
    encode_row,
    load_rows,
)


@torch.no_grad()
def predict_trained(model, tok, rows, device, batch_questions=16, max_tokens=8192):
    preds = {}
    for i in range(0, len(rows), batch_questions):
        chunk = rows[i : i + batch_questions]
        paths, spans = [], []
        for row in chunk:
            _, _, rp = encode_row(tok, row, max_tokens)
            spans.append((len(paths), len(rp)))
            paths.extend(rp)
        tokens, mask, lengths = collate(paths, tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(tokens, mask, lengths)
        for row, (off, k) in zip(chunk, spans):
            preds[row["id"]] = F.softmax(scores[off : off + k], dim=-1).float().cpu()
    return preds


@torch.no_grad()
def predict_zero_shot(tok, rows, device, batch_questions=8):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    preds = {}
    for i in range(0, len(rows), batch_questions):
        chunk = rows[i : i + batch_questions]
        prompts, label_ids = [], []
        for row in chunk:
            p, lids, _ = encode_row(tok, row)
            prompts.append(p)
            label_ids.append(lids)
        tokens, mask, lengths = collate(prompts, tok.pad_token_id, device)
        logits = model(input_ids=tokens, attention_mask=mask).logits
        last = logits[torch.arange(len(chunk), device=device), lengths - 1].float()
        for j, row in enumerate(chunk):
            z = last[j, torch.tensor(label_ids[j], device=device)]
            preds[row["id"]] = F.softmax(z, dim=-1).cpu()
    del model
    torch.cuda.empty_cache()
    return preds


def row_metrics(p, t):
    eps = 1e-12
    return {
        "acc": float(p.argmax().item() == t.argmax().item()),
        "nll": float(-(t * (p + eps).log()).sum()),
        "brier": float(((p - t) ** 2).sum()),
        "tv": float(0.5 * (p - t).abs().sum()),
        "conf": float(p.max()),
    }


def ece(items, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for m in items:
        b = min(int(m["conf"] * n_bins), n_bins - 1)
        bins[b].append(m)
    total = len(items)
    return sum(
        len(b) / total * abs(sum(x["acc"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b))
        for b in bins if b
    )


def summarize(items):
    n = len(items)
    return {
        "n": n,
        "accuracy": sum(m["acc"] for m in items) / n,
        "nll": sum(m["nll"] for m in items) / n,
        "brier": sum(m["brier"] for m in items) / n,
        "tv": sum(m["tv"] for m in items) / n,
        "ece10": ece(items),
    }


def paired_consistency(rows, preds):
    by_id = {r["id"]: r for r in rows}
    diffs, seen = [], set()
    for r in rows:
        if r["primitive"] != "choice" or not r["paired_id"] or r["id"] in seen:
            continue
        partner = by_id.get(r["paired_id"])
        if partner is None or partner["primitive"] != "noul":
            continue
        if r["id"] not in preds or partner["id"] not in preds:
            continue
        p_choice_first = float(preds[r["id"]][0])
        yes_idx = partner["candidates"].index("yes")
        p_yes = float(preds[partner["id"]][yes_idx])
        diffs.append(abs(p_choice_first - p_yes))
        seen.update({r["id"], partner["id"]})
    return {"n_pairs": len(diffs), "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--batch-questions", type=int, default=16)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if not args.zero_shot and not args.ckpt:
        ap.error("specify --zero-shot or --ckpt DIR")

    device = "cuda"
    rows = load_rows(args.split, args.data)
    if args.n:
        import random

        rows = random.Random(0).sample(rows, args.n)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    if args.zero_shot:
        preds = predict_zero_shot(tok, rows, device, args.batch_questions)
        tag = "zero_shot"
    else:
        model = StudentModel().to(device)
        from train import load_ckpt

        load_ckpt(model, args.ckpt)
        model.eval()
        preds = predict_trained(model, tok, rows, device, args.batch_questions)
        tag = Path(args.ckpt).name

    items, groups = [], defaultdict(list)
    for row in rows:
        if row["id"] not in preds:
            continue
        t = torch.tensor([row["target_distribution"][c] for c in row["candidates"]])
        m = row_metrics(preds[row["id"]], t)
        m["id"] = row["id"]
        items.append(m)
        groups[(row["mechanism"], row["difficulty"])].append(m)

    report = {
        "tag": tag,
        "split": args.split,
        "overall": summarize(items),
        "by_mechanism_difficulty": {
            f"{m}/{d}": summarize(v) for (m, d), v in sorted(groups.items())
        },
        "paired_consistency": paired_consistency(rows, preds),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        with open(out.with_suffix(".preds.jsonl"), "w") as f:
            for row in rows:
                if row["id"] in preds:
                    f.write(json.dumps({
                        "id": row["id"],
                        "probs": [round(float(x), 6) for x in preds[row["id"]]],
                    }) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
