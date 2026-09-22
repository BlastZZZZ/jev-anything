"""Run a policy on NanoJev's frozen 548-case cohort with their seeded controller.

Reuses NanoJev's own environments and controller (epsilon-greedy, seed 17,
lexicographic tie-break) so results are directly comparable to the published
selected/jev/native trajectories. Engines: vcdm (internal codename for the
xiaojev checkpoints), nanojev (their DecisionPredictor + public weights).

Environment variables:
  XIAOJEV_NANOJEV_REPO   local clone of https://github.com/TianyuCodings/NanoJev
  XIAOJEV_NANOJEV_DATA   NanoJev-Data dataset root (contains evaluation/)
  XIAOJEV_NANOJEV_CKPT   NanoJev weights path or HF id (default C-Tianyu/NanoJev)
  XIAOJEV_CKPT_V1        xiaojev v1 checkpoint (default <repo>/ckpt/v1)
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

_NANOJEV_DATA = os.environ.get("XIAOJEV_NANOJEV_DATA")
CASES = (
    str(Path(_NANOJEV_DATA) / "evaluation" / "test_cases.jsonl")
    if _NANOJEV_DATA
    else None
)
NANOJEV_CKPT = os.environ.get("XIAOJEV_NANOJEV_CKPT", "C-Tianyu/NanoJev")
VCDM_CKPT = os.environ.get("XIAOJEV_CKPT_V1", str(REPO_ROOT / "ckpt" / "v1"))


def nanojev_scripts():
    env = os.environ.get("XIAOJEV_NANOJEV_REPO")
    p = Path(env) / "scripts" if env else None
    if p is None or not p.is_dir():
        raise SystemExit(
            "NanoJev scripts not found. Clone "
            "https://github.com/TianyuCodings/NanoJev and set "
            "XIAOJEV_NANOJEV_REPO to the clone."
        )
    return p


class VCDMPredictor:
    def __init__(self, ckpt=VCDM_CKPT):
        import torch
        from transformers import AutoTokenizer
        from train import MODEL_PATH, StudentModel, load_ckpt, encode_row, collate
        self._torch = torch
        self._encode_row = encode_row
        self._collate = collate
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = StudentModel().to("cuda")
        load_ckpt(self.model, ckpt)
        self.model.eval()

    def predict(self, payload, batch_questions=16, temperature=1.0):
        torch = self._torch
        requests = payload["states"]
        outputs = {r["id"]: {"id": r["id"], "answers": {}} for r in requests}
        for off in range(0, len(requests), batch_questions):
            chunk = requests[off : off + batch_questions]
            paths, spans = [], []
            for req in chunk:
                q = req["questions"]["action"]
                row = {"primitive": "choice", "state": req["state"],
                       "instruction": q["instructions"],
                       "candidates": [q["criteria"][k] for k in q["criteria"]]}
                _, _, rp = self._encode_row(self.tok, row)
                spans.append((len(paths), list(q["criteria"])))
                paths.extend(rp)
            tokens, mask, lengths = self._collate(paths, self.tok.pad_token_id, "cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                scores = self.model(tokens, mask, lengths)
            for req, (o, aids) in zip(chunk, spans):
                z = scores[o : o + len(aids)].double() / temperature
                probs = torch.softmax(z, dim=-1)
                total = probs.sum()
                probs = (probs / total).tolist()  # exact unit sum for the verifier
                outputs[req["id"]]["answers"]["action"] = {
                    "probabilities": {a: p for a, p in zip(aids, probs)}}
        return {"states": list(outputs.values())}


class NanoJevPredictor:
    def __init__(self, ckpt=NANOJEV_CKPT):
        sys.path.insert(0, str(nanojev_scripts()))
        from predict_toy_decisions import DecisionPredictor
        self.inner = DecisionPredictor(ckpt, max_length=8192, disable_native_triton=False)

    def predict(self, payload, batch_questions=16, temperature=1.0):
        result = self.inner.predict(payload, batch_questions=batch_questions, temperature=temperature)
        for state in result["states"]:
            answer = state["answers"]["action"]
            probs = answer["probabilities"]
            total = math.fsum(probs.values())
            answer["probabilities"] = {k: v / total for k, v in probs.items()}
        return result


def rollout(args):
    sys.path.insert(0, str(nanojev_scripts()))
    from unified_game_pipeline import (  # noqa: E402
        LocalEnvironments, behavior_distribution, choose, digest, encode,
        file_digest, read_rows, validate_cases, write_json,
    )
    cases = read_rows(args.cases)
    validate_cases(cases)
    if args.limit:
        counts, selected = Counter(), []
        for case in cases:
            key = (case["split"], case["variant"])
            if counts[key] < args.limit:
                counts[key] += 1
                selected.append(case)
        cases = selected
    predictor = {"vcdm": VCDMPredictor, "nanojev": NanoJevPredictor}[args.engine](args.checkpoint) \
        if args.checkpoint else {"vcdm": VCDMPredictor, "nanojev": NanoJevPredictor}[args.engine]()
    policy = {"engine": args.engine, "controller": "greedy", "epsilon": 0.1, "sampling_seed": 17,
              "temperature": 1.0, "tie_break": "lexicographic_first",
              "environment_contract": "finite_task_deadline_v1",
              "checkpoint": str(args.checkpoint or {"vcdm": VCDM_CKPT, "nanojev": NANOJEV_CKPT}[args.engine])}
    policy_id = digest(policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("fresh output path required")
    all_episodes = []
    started = time.monotonic()
    with output.open("x") as handle:
        for offset in range(0, len(cases), args.env_batch):
            batch = cases[offset : offset + args.env_batch]
            environments = LocalEnvironments()
            try:
                current = environments.reset(batch)
                episodes = {c["id"]: {"case": c, "continuation_policy_id": policy_id, "steps": [],
                                      "complete": False, "success": None} for c in batch}
                rngs = {c["id"]: random.Random(int(digest([c["id"], 17])[:16], 16)) for c in batch}
                active = set(episodes)
                while active:
                    requests, decisions = [], {}
                    for key in sorted(active):
                        obs = current[key]["observation"]
                        candidates = obs["candidates"]
                        if not candidates:
                            episodes[key].update(complete=True, success=bool(current[key]["info"]["success"]),
                                                 final_info=current[key]["info"])
                            continue
                        if len(candidates) == 1:
                            decisions[key] = {"scores": {next(iter(candidates)): 1.0}, "answers": {}, "forced": True}
                        else:
                            requests.append({"id": key, "state": obs["state"], "questions": {"action": {
                                "type": "choice",
                                "instructions": "Choose the next action that maximizes the probability of completing the stated task successfully before its deadline. Use the visible state, action descriptions, remaining time, and recorded history.",
                                "criteria": candidates}}})
                    active = {k for k in active if not episodes[k]["complete"]}
                    if requests:
                        response = predictor.predict({"states": requests}, batch_questions=args.batch_questions)
                        answers_by_id = {r["id"]: r["answers"] for r in response["states"]}
                        for req in requests:
                            answers = answers_by_id[req["id"]]
                            decisions[req["id"]] = {"scores": answers["action"]["probabilities"], "answers": answers}
                    actions = {}
                    for key in sorted(active):
                        d = decisions[key]
                        d["behavior_probs"] = behavior_distribution(d["scores"], "greedy", 0.1)
                        actions[key] = choose(d["behavior_probs"], rngs[key])
                    if not actions:
                        break
                    following = environments.step(actions)
                    for key, action in actions.items():
                        t = following[key]
                        episodes[key]["steps"].append({"observation": current[key]["observation"],
                            "action": action, **decisions[key], "reward": t["reward"],
                            "terminated": t["terminated"], "truncated": t["truncated"], "info": t["info"]})
                        if t["truncated"]:
                            raise RuntimeError("external truncation")
                        if t["terminated"]:
                            episodes[key].update(complete=True, success=bool(t["info"]["success"]),
                                                 final_info=t["info"])
                            active.remove(key)
                    current.update(following)
                for case in batch:
                    handle.write(encode(episodes[case["id"]]) + "\n")
                    handle.flush()
                    all_episodes.append(episodes[case["id"]])
                print(encode({"completed": len(all_episodes), "total": len(cases),
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
            finally:
                environments.close()
    manifest = {"schema_version": "vcdm-frozen-cohort-run-v1", "policy": policy,
                "continuation_policy_id": policy_id, "cases_sha256": file_digest(args.cases),
                "selected_cases": [c["id"] for c in cases], "finished": True,
                "episode_sha256": file_digest(output),
                "elapsed_seconds": time.monotonic() - started}
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(encode({"done": len(all_episodes), "output": str(output)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vcdm", "nanojev"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cases", default=CASES,
                    help="test_cases.jsonl from NanoJev-Data "
                         "(default: $XIAOJEV_NANOJEV_DATA/evaluation/test_cases.jsonl)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--env-batch", type=int, default=16)
    ap.add_argument("--batch-questions", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="per split/variant cap, 0=all")
    args = ap.parse_args()
    if not args.cases:
        ap.error("--cases not given and XIAOJEV_NANOJEV_DATA is not set")
    rollout(args)


if __name__ == "__main__":
    main()
