"""Latency benchmark: xiaojev v2 / v1 / NanoJev public weights.

1. Single-question end-to-end latency (tokenize + forward + group softmax),
   p50/p95 on 100 probability questions and 100 game questions.
2. Packed 24-question batch throughput, end-to-end and warm-forward-only
   (the latter matches NanoJev's published A100 protocol: 8 states, 24
   questions, 68 candidate paths, <=85 tokens, forward only, p50 84.72ms,
   ~283 questions/s -- ours is a single RTX 3090).

Environment variables:
  XIAOJEV_NANOJEV_REPO   local clone of https://github.com/TianyuCodings/NanoJev
  XIAOJEV_NANOJEV_CKPT   NanoJev weights path or HF id (default C-Tianyu/NanoJev)
  XIAOJEV_CKPT_V1 / XIAOJEV_CKPT_V2   xiaojev checkpoints (default <repo>/ckpt/v1|v2)
  XIAOJEV_PROB_DATA      probability-task JSONL (default <repo>/data/train_v1.jsonl)
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from train import LABELS, MODEL_PATH, StudentModel, collate, encode_row, load_ckpt, load_rows  # noqa: E402

OUT = REPO_ROOT / "results" / "latency_bench.json"
CKPT_V1 = os.environ.get("XIAOJEV_CKPT_V1", str(REPO_ROOT / "ckpt" / "v1"))
CKPT_V2 = os.environ.get("XIAOJEV_CKPT_V2", str(REPO_ROOT / "ckpt" / "v2"))
NANOJEV_CKPT = os.environ.get("XIAOJEV_NANOJEV_CKPT", "C-Tianyu/NanoJev")
GAMES_DATA = os.environ.get("XIAOJEV_GAME_DATA", str(REPO_ROOT / "data" / "games_v1.jsonl"))
NANOJEV_REF = {"gpu": "A100-SXM4-80GB", "p50_ms": 84.72, "questions_per_second": 283,
               "scope": "8 states, 24 questions, 68 candidate paths, <=85 tokens, warm forward only"}


def nanojev_scripts():
    env = os.environ.get("XIAOJEV_NANOJEV_REPO")
    p = Path(env) / "scripts" if env else None
    if p is None or not p.is_dir():
        raise SystemExit(
            "NanoJev scripts not found. Clone "
            "https://github.com/TianyuCodings/NanoJev and set "
            "XIAOJEV_NANOJEV_REPO to the clone (or use --skip-nanojev)."
        )
    return p


def pct(vals, frac):
    vals = sorted(vals)
    i = (len(vals) - 1) * frac
    lo, hi = int(i), min(int(i) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (i - lo)


def stats(lat_ms, n_questions):
    return {"p50_ms": round(pct(lat_ms, 0.5), 2), "p95_ms": round(pct(lat_ms, 0.95), 2),
            "mean_ms": round(sum(lat_ms) / len(lat_ms), 2),
            "questions_per_second_from_p50": round(1000 * n_questions / pct(lat_ms, 0.5), 1)}


class VCDMEngine:
    def __init__(self, ckpt):
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = StudentModel().to("cuda")
        load_ckpt(self.model, ckpt)
        self.model.eval()

    def _paths(self, rows):
        paths, spans = [], []
        for row in rows:
            _, _, rp = encode_row(self.tok, row)
            spans.append(len(rp))
            paths.extend(rp)
        return paths, spans

    def single(self, row):
        t0 = time.perf_counter()
        paths, _ = self._paths([row])
        tokens, mask, lengths = collate(paths, self.tok.pad_token_id, "cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            scores = self.model(tokens, mask, lengths)
        torch.softmax(scores, dim=-1)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    def batch_end_to_end(self, rows):
        t0 = time.perf_counter()
        paths, _ = self._paths(rows)
        tokens, mask, lengths = collate(paths, self.tok.pad_token_id, "cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            scores = self.model(tokens, mask, lengths)
        off = 0
        for row in rows:
            k = len(row["candidates"])
            torch.softmax(scores[off : off + k], dim=-1)
            off += k
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    def batch_forward_only(self, rows):
        paths, _ = self._paths(rows)
        tokens, mask, lengths = collate(paths, self.tok.pad_token_id, "cuda")

        def once():
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.model(tokens, mask, lengths)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000
        return once, len(paths), max(lengths.tolist())


def to_nanojev_request(row):
    if row["primitive"] == "choice":
        q = {"action": {"type": "choice", "instructions": row["instruction"],
                        "criteria": {LABELS[i]: c for i, c in enumerate(row["candidates"])}}}
    else:
        q = {"verdict": {"type": "boolean",
                         "instructions": "Judge whether the following statement about the state is true.\n\nSTATEMENT:\n" + row["proposition"],
                         "criteria": {"false": "The statement is false.", "true": "The statement is true."}}}
    return {"id": row["id"], "state": row["state"], "questions": q}


class NanoJevEngine:
    def __init__(self, ckpt):
        sys.path.insert(0, str(nanojev_scripts()))
        from predict_toy_decisions import DecisionPredictor
        self.inner = DecisionPredictor(ckpt, max_length=8192, disable_native_triton=False)

    def single(self, row):
        t0 = time.perf_counter()
        self.inner.predict({"states": [to_nanojev_request(row)]}, batch_questions=0)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    def batch_end_to_end(self, rows):
        t0 = time.perf_counter()
        self.inner.predict({"states": [to_nanojev_request(r) for r in rows]}, batch_questions=0)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-v2", default=CKPT_V2, help="xiaojev v2 checkpoint (default: %(default)s)")
    ap.add_argument("--ckpt-v1", default=CKPT_V1, help="xiaojev v1 checkpoint (default: %(default)s)")
    ap.add_argument("--nanojev-ckpt", default=NANOJEV_CKPT,
                    help="NanoJev weights path or HF id (default: %(default)s)")
    ap.add_argument("--skip-nanojev", action="store_true",
                    help="benchmark only the xiaojev checkpoints")
    ap.add_argument("--data", default=None,
                    help="probability JSONL (default: $XIAOJEV_PROB_DATA or <repo>/data/train_v1.jsonl)")
    ap.add_argument("--games-data", default=GAMES_DATA,
                    help="game JSONL (default: %(default)s)")
    ap.add_argument("--n", type=int, default=100, help="single-question sample size per domain")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    random.seed(0)
    rng = random.Random(0)
    prob_rows_all = load_rows("test", args.data) if args.data else load_rows("test")
    prob_rows = rng.sample(prob_rows_all, args.n)
    game_rows = rng.sample(load_rows("test", args.games_data), args.n)
    prob_pack = prob_rows[:24]
    game_pack = game_rows[:24]

    engines = [
        ("vcdm_v2", VCDMEngine(args.ckpt_v2)),
        ("vcdm_v1", VCDMEngine(args.ckpt_v1)),
    ]
    if not args.skip_nanojev:
        engines.append(("nanojev", NanoJevEngine(args.nanojev_ckpt)))
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "parameter_storage": "float32", "forward_autocast": "bfloat16",
              "single_scope": "tokenize + forward + group softmax, one question at a time, warm",
              "nanojev_a100_reference": NANOJEV_REF, "engines": {}}
    for name, eng in engines:
        rep = {}
        for domain, rows in (("probability", prob_rows), ("game", game_rows)):
            for r in rows[:5]:  # warmup
                eng.single(r)
            lat = [eng.single(r) for r in rows]
            rep[f"single_{domain}"] = stats(lat, 1) | {"n": len(rows)}
        for domain, pack in (("probability", prob_pack), ("game", game_pack)):
            eng.batch_end_to_end(pack)
            eng.batch_end_to_end(pack)
            lat = [eng.batch_end_to_end(pack) for _ in range(20)]
            entry = {"end_to_end": stats(lat, len(pack)), "n_questions": len(pack),
                     "n_candidate_paths": sum(len(r["candidates"]) for r in pack)}
            if hasattr(eng, "batch_forward_only"):
                once, n_paths, max_len = eng.batch_forward_only(pack)
                once(); once()
                lat_f = [once() for _ in range(20)]
                entry["forward_only"] = stats(lat_f, len(pack)) | {
                    "n_candidate_paths": n_paths, "max_path_tokens": max_len}
            rep[f"batch24_{domain}"] = entry
        report["engines"][name] = rep
        print(name, json.dumps(rep, indent=1), flush=True)
        if name != "nanojev":
            del eng.model
        del eng
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
