"""Adapt the last six v4 layers to browser pages; select using fixed dev rows."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("XIAOJEV_BROWSER_RUN", str(ROOT / "results/v4_browser")))
sys.path.insert(0, str(ROOT / "training"))
os.environ.update(
    OMP_NUM_THREADS="4",
    TOKENIZERS_PARALLELISM="false",
    PYTORCH_ALLOC_CONF="expandable_segments:True",
)
import torch
from evaluate import predict_trained
from train import (
    MODEL_PATH,
    StudentModel,
    collate,
    encode_row,
    load_ckpt,
    load_rows,
    microbatch,
    save_ckpt,
)
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(OUT / "browser_mixed.jsonl"))
    parser.add_argument("--init", default=str(ROOT / "ckpt/v4"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=75)
    parser.add_argument("--dev-n", type=int, default=300)
    parser.add_argument("--name", default="v4_browser_repair")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    torch.manual_seed(20260924)
    rng = random.Random(20260924)
    torch.set_num_threads(4)
    rows = load_rows("train", args.data)
    dev = random.Random(17).sample(load_rows("dev", args.data), args.dev_n)
    assert not {r["meta"]["scenario_id"] for r in rows} & {
        r["meta"]["scenario_id"] for r in dev
    }
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = StudentModel().to("cuda")
    load_ckpt(model, args.init)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for layer in model.backbone.layers[-6:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in model.backbone.norm.parameters():
        p.requires_grad_(True)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 2e-5},
            {
                "params": list(model.norm.parameters()) + list(model.head.parameters()),
                "lr": 1e-4,
            },
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 15, args.steps)
    print(
        "trainable parameters",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        flush=True,
    )
    best = -1
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            preds = predict_trained(model, tok, dev, "cuda", batch_questions=2)
            acc = sum(
                int(preds[r["id"]].argmax())
                == max(
                    range(len(r["candidates"])),
                    key=lambda i: r["target_distribution"][r["candidates"][i]],
                )
                for r in dev
            ) / len(dev)
            print(json.dumps({"dev_step": step, "accuracy": acc}), flush=True)
            with (OUT / f"{args.prefix}browser_dev.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "accuracy": acc}) + "\n")
            if acc > best:
                best = acc
                path = (
                    ROOT / f"ckpt/{args.name}/step{step}" if step else Path(args.init)
                )
                if step:
                    save_ckpt(
                        model, optimizer, scheduler, step, path, save_optimizer=False
                    )
                (OUT / f"{args.prefix}browser_selected.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": str(path),
                            "step": step,
                            "selection": "highest accuracy on fixed dev, ties prefer earlier checkpoint",
                            "dev_accuracy": acc,
                            "dev_n": len(dev),
                            "data": args.data,
                            "initialized_from": args.init,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            model.train()
        if step == args.steps:
            break
        started = time.monotonic()
        batch = rng.sample(rows, 8)
        paths, slices, targets = [], [], []
        for row in batch:
            _, _, rp = encode_row(tok, row)
            slices.append((len(paths), len(rp)))
            paths.extend(rp)
            targets.append(
                torch.tensor(
                    [row["target_distribution"][c] for c in row["candidates"]],
                    device="cuda",
                )
            )
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0
        for indices in microbatch(slices, paths, 8192):
            mb = [
                paths[j] for i in indices for j in range(slices[i][0], sum(slices[i]))
            ]
            tokens, mask, lengths = collate(mb, tok.pad_token_id, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(tokens, mask, lengths)
            offset = 0
            loss = scores.new_zeros(())
            for i in indices:
                k = slices[i][1]
                lp = torch.log_softmax(scores[offset : offset + k], dim=-1)
                offset += k
                loss += (
                    -(targets[i] * lp).sum()
                    + 0.1 * ((lp.exp() - targets[i]) ** 2).sum()
                )
            (loss / len(batch)).backward()
            loss_value += float(loss.detach()) / len(batch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        record = {
            "step": step + 1,
            "loss": loss_value,
            "elapsed_s": time.monotonic() - started,
        }
        with (OUT / f"{args.prefix}train_browser.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if (step + 1) % 10 == 0:
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
