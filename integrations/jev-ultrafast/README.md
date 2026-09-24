# jev anything backend for jev-ultrafast

The adapter chooses operations and observed element indices with jev anything. A separate OpenAI-compatible text helper supplies only `TYPE_TEXT` values. Decisions return the TypeSafe-compatible response shape with zero autoregressive decoding; large candidate sets may require multiple forward microbatches.

The integration includes the state-serialization repair, shared training/inference request construction, and a progress guard that ignores transient DOM node identities. Copying `local_model.py` alone is insufficient: the dispatch and progress changes in `agent.patch` are also required.

## Install

Run from the jev anything repository root:

```bash
git clone https://github.com/browser-use/jev-ultrafast.git ../jev-ultrafast
git -C ../jev-ultrafast checkout 1231850a0bf1a0c0341fe408ef1668dbbfdfac46
python integrations/jev-ultrafast/install.py ../jev-ultrafast
```

The upstream commit is recorded in `upstream.json`. The installer checks that the patch applies before changing tracked files and also recognizes an already-applied patch. It installs the backend and regression tests. Upstream code remains under its MIT license.

Install upstream dependencies in a Python 3.12 environment, plus PyTorch and Transformers from this repository's `requirements.txt`. CUDA and the base tokenizer are required. Place the separately trained browser weights at `ckpt/v4_browser`, or set `XIAOJEV_CKPT` explicitly.

```bash
export XIAOJEV_HOME="$PWD"
export XIAOJEV_JEV_HOME="$PWD/../jev-ultrafast"
export XIAOJEV_CKPT="$PWD/ckpt/v4_browser"
export JEV_BACKEND=local
# Set CUDA_VISIBLE_DEVICES to the GPU assigned to your run.
export BU_CDP_URL=http://127.0.0.1:9222
export TEXT_MODEL_BASE_URL=http://127.0.0.1:8020/v1
export TEXT_MODEL=qwen3.8-27b
export TEXT_MODEL_API_KEY=local-dummy
export TEXT_MODEL_REASONING=none
python experiments/v4_browser/browser_acceptance.py --tag local
```

Chrome CDP, the upstream fixture server on port 8766, and the local text helper must already be running. Configure the base tokenizer/model using `XIAOJEV_BASE_MODEL` when needed. A TypeSafe API key is unnecessary for this backend. Actual final outcomes still need independent verification.

## Verified scope

Original v4 failed both repeated hotel tasks, repeatedly clicking Find stays. The repaired `v4_browser_dom/step100` passed all four local regressions twice: the original task repeated, another city, and another category with free cancellation off. Final runs used 5, 5, 5, and 4 actions, with one text-helper call each.

Adaptation uses rendered states from the same fixture layout, and these cases participate in deployment selection. This is a local regression result, not an independent general-web benchmark. Higher-dev step150 was rejected after passing only 3/4; successful and rejected traces are retained. Browser-adapted weights have not been evaluated as a replacement in other domains.

See [training and acceptance](../../experiments/v4_browser/README.md), [the release report](../../docs/V4_REPAIR.md), and [raw final results](../../results/v4_repair/browser_final_summary.json).
