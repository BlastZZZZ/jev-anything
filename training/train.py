"""xiaojev: train a calibrated decision model on programmatic probability data.

Qwen3-0.6B backbone + LayerNorm + scalar head on the EOS-position hidden state
of each (prompt, label, EOS) candidate path; per-question softmax + soft-label CE.

Configuration (environment variables):
  XIAOJEV_BASE_MODEL  backbone path or HF id (default Qwen/Qwen3-0.6B)
  XIAOJEV_PROB_DATA   probability-task JSONL (default <repo>/data/train_v1.jsonl)
"""

import argparse
import hashlib
import json
import os
import random
import string
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = os.environ.get("XIAOJEV_BASE_MODEL", "Qwen/Qwen3-0.6B")
DATA_PATH = os.environ.get(
    "XIAOJEV_PROB_DATA", str(REPO_ROOT / "data" / "train_v1.jsonl")
)
LABELS = list(string.digits + string.ascii_uppercase)


def load_rows(split, path=DATA_PATH):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["split"] == split:
                rows.append(d)
    return rows


def labels_for(row):
    if row["primitive"] == "choice":
        return LABELS[: len(row["candidates"])]
    return list(row["candidates"])  # yes / no


def build_messages(row):
    if row["primitive"] == "choice":
        table = "\n".join(f"[{LABELS[i]}] {c}" for i, c in enumerate(row["candidates"]))
        text = (
            "You are a decision system. Read the state, then answer the question by "
            "choosing exactly one option. Judge the true probability of each option "
            "from the state; do not default to uniform and do not be overconfident.\n\n"
            f"STATE:\n{row['state']}\n\nQUESTION:\n{row['instruction']}\n\nOPTIONS:\n{table}\n\n"
            "Answer with only the option label."
        )
    else:
        text = (
            "You are a decision system. Read the state, then judge whether the "
            "statement is true. Answer with only yes or no. Judge the true "
            "probability; do not be overconfident.\n\n"
            f"STATE:\n{row['state']}\n\nSTATEMENT:\n{row['proposition']}\n\n"
            "Answer with only yes or no."
        )
    return [{"role": "user", "content": text}]


def encode_row(tok, row, max_tokens=8192):
    text = tok.apply_chat_template(
        build_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tok.encode(text, add_special_tokens=False)
    budget = max_tokens - 2  # label token + EOS
    state = row["state"]
    while len(prompt_ids) > budget and len(state) > 16:
        state = state[: int(len(state) * 0.8)]
        text = tok.apply_chat_template(
            build_messages({**row, "state": state + " ..."}),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = tok.encode(text, add_special_tokens=False)
    label_ids = []
    for l in labels_for(row):
        ids = tok.encode(l, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {l!r} is not a single token: {ids}")
        label_ids.append(ids[0])
    paths = [prompt_ids + [l, tok.eos_token_id] for l in label_ids]
    return prompt_ids, label_ids, paths


class StudentModel(nn.Module):
    def __init__(self, model_path=MODEL_PATH, backbone=None):
        super().__init__()
        if backbone is None:
            backbone = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.float32, attn_implementation="sdpa"
            )
        backbone.config.use_cache = False
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, lengths):
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        leaf = hs[torch.arange(input_ids.size(0), device=hs.device), lengths - 1]
        return self.head(self.norm(leaf)).squeeze(-1).float()


def collate(paths, pad_id, device):
    lengths = torch.tensor([len(p) for p in paths], device=device)
    width = int(lengths.max())
    tokens = torch.full((len(paths), width), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(paths):
        tokens[i, : len(p)] = torch.tensor(p, device=device)
    mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    return tokens, mask, lengths


def microbatch(row_slices, paths, budget):
    """Pack rows (a row's candidate paths must stay in one forward for the
    group softmax) into microbatches by padded-token budget."""
    order = sorted(range(len(row_slices)), key=lambda r: -len(paths[row_slices[r][0]]))
    batches, cur, cur_max = [], [], 0
    for r in order:
        start, k = row_slices[r]
        length = len(paths[start])
        new_max = max(cur_max, length)
        new_count = sum(row_slices[x][1] for x in cur) + k
        if cur and new_max * new_count > budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(r)
        cur_max = max(cur_max, length)
    if cur:
        batches.append(cur)
    return batches


def group_losses(scores, sizes, targets, brier_mu):
    ce_rows, brier_rows = [], []
    offset = 0
    for size, t in zip(sizes, targets):
        z = scores[offset : offset + size]
        logp = F.log_softmax(z, dim=-1)
        ce_rows.append(-(t * logp).sum())
        if brier_mu > 0:
            brier_rows.append(((logp.exp() - t) ** 2).sum())
        offset += size
    ce = torch.stack(ce_rows).mean()
    brier = torch.stack(brier_rows).mean() if brier_rows else scores.new_zeros(())
    return ce + brier_mu * brier, torch.stack([c.detach() for c in ce_rows]), brier.detach()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_ckpt(model, optimizer, scheduler, step, outdir, save_optimizer=True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backbone_dir = outdir / "backbone"
    model.backbone.save_pretrained(backbone_dir)
    torch.save({"norm": model.norm.state_dict(), "head": model.head.state_dict()}, outdir / "head.pt")
    trainer = {"step": step, "scheduler": scheduler.state_dict()}
    if save_optimizer:
        trainer["optimizer"] = optimizer.state_dict()
    torch.save(trainer, outdir / "trainer.pt")
    manifest = {}
    for p in sorted(outdir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest[str(p.relative_to(outdir))] = sha256_file(p)
    with open(outdir / "manifest.json", "w") as f:
        json.dump({"step": step, "sha256": manifest}, f, indent=2)


def load_ckpt(model, ckpt_dir, optimizer=None, scheduler=None):
    ckpt_dir = Path(ckpt_dir)
    from transformers import AutoModel as _AM
    model.backbone = _AM.from_pretrained(
        ckpt_dir / "backbone", torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.backbone.config.use_cache = False
    model.backbone.to(model.head.weight.device)
    head = torch.load(ckpt_dir / "head.pt", map_location="cpu", weights_only=True)
    model.norm.load_state_dict(head["norm"])
    model.head.load_state_dict(head["head"])
    trainer = torch.load(ckpt_dir / "trainer.pt", map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in trainer:
        optimizer.load_state_dict(trainer["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(trainer["scheduler"])
    return trainer["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-questions", type=int, default=24)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--brier-mu", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "ckpt" / "v1"))
    ap.add_argument("--log", default=str(REPO_ROOT / "results" / "train_log.jsonl"))
    ap.add_argument("--data", default=DATA_PATH, help="primary JSONL data source")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--mix", default=None, help="second JSONL data source (train split)")
    ap.add_argument("--mix-ratio", type=float, default=0.5,
                    help="fraction of each step's questions drawn from --mix")
    ap.add_argument("--microbatch-tokens", type=int, default=32768)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    rows = load_rows("train", args.data)
    print(f"train rows: {len(rows)}")
    mix_rows = load_rows("train", args.mix) if args.mix else None
    if mix_rows:
        print(f"mix rows: {len(mix_rows)} (ratio {args.mix_ratio})")

    model = StudentModel().to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    head_params = list(model.norm.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ]
    )
    warmup = max(1, int(args.steps * args.warmup_frac))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, args.steps)

    start_step = 0
    if args.resume:
        start_step = load_ckpt(model, args.resume, optimizer, scheduler)
        model.to(device)
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print(f"resumed from {args.resume} at step {start_step}")

    model.train()
    log_f = open(args.log, "a")
    for step in range(start_step, args.steps):
        t0 = time.perf_counter()
        if mix_rows:
            n_mix = round(args.batch_questions * args.mix_ratio)
            batch = random.sample(rows, args.batch_questions - n_mix) + random.sample(mix_rows, n_mix)
            domains = [0] * (args.batch_questions - n_mix) + [1] * n_mix
        else:
            batch = random.sample(rows, args.batch_questions)
            domains = [0] * args.batch_questions
        paths, sizes, targets = [], [], []
        row_slices = []
        for row in batch:
            _, _, row_paths = encode_row(tok, row, args.max_tokens)
            row_slices.append((len(paths), len(row_paths)))
            paths.extend(row_paths)
            sizes.append(len(row_paths))
            targets.append(
                torch.tensor([row["target_distribution"][c] for c in row["candidates"]],
                             dtype=torch.float32, device=device)
            )
        optimizer.zero_grad(set_to_none=True)
        ce_vals = [None] * len(batch)
        brier_sum = 0.0
        for mb_rows in microbatch(row_slices, paths, args.microbatch_tokens):
            mb_paths = [paths[i] for r in mb_rows
                        for i in range(row_slices[r][0], row_slices[r][0] + row_slices[r][1])]
            tokens, mask, lengths = collate(mb_paths, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mb_scores = model(tokens, mask, lengths)
            mb_loss = mb_scores.new_zeros(())
            off = 0
            for r in mb_rows:
                k = row_slices[r][1]
                z = mb_scores[off : off + k]
                off += k
                logp = F.log_softmax(z, dim=-1)
                t = targets[r]
                ce = -(t * logp).sum()
                ce_vals[r] = ce.item()
                mb_loss = mb_loss + ce
                if args.brier_mu > 0:
                    br = ((logp.exp() - t) ** 2).sum()
                    brier_sum += br.item()
                    mb_loss = mb_loss + args.brier_mu * br
            (mb_loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        dt = time.perf_counter() - t0
        ce_by_domain = {}
        for d in set(domains):
            vals = [ce_vals[i] for i, x in enumerate(domains) if x == d and ce_vals[i] is not None]
            ce_by_domain[d] = sum(vals) / len(vals)
        rec = {
            "step": step + 1,
            "loss": round(sum(ce_vals) / len(ce_vals) + args.brier_mu * brier_sum / len(batch), 5),
            "ce": round(sum(ce_vals) / len(ce_vals), 5),
            "ce_prob": round(ce_by_domain.get(0, float("nan")), 5),
            "ce_game": round(ce_by_domain.get(1, float("nan")), 5),
            "brier": round(brier_sum / len(batch), 5),
            "lr_backbone": scheduler.get_last_lr()[0],
            "lr_head": scheduler.get_last_lr()[1],
            "n_paths": len(paths),
            "step_time_s": round(dt, 3),
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(rec)
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(model, optimizer, scheduler, step + 1, Path(args.out) / f"step{step+1}")
    log_f.close()

    save_ckpt(model, optimizer, scheduler, args.steps, args.out)
    print(f"saved final checkpoint to {args.out}")


if __name__ == "__main__":
    main()
