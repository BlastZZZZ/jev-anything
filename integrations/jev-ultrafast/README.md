# jev-ultrafast integration — xiaojev as a local browser-agent backend

`local_model.py` plugs xiaojev into
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT): the
browser agent's decision step calls this module instead of the paid TypeSafe
API. The response shape is identical to the API (`answers` per question with
`choice` / `probabilities` / `confidence`, plus `model` and `usage`), so no
other agent code changes. One packed forward pass scores all candidates of all
questions; the model only ever returns an index into the offered candidates —
it never emits selectors, code, or text.

## Wiring

```bash
# 1. clone the agent
git clone https://github.com/browser-use/jev-ultrafast
cd jev-ultrafast

# 2. drop this backend into the package (overwrites nothing upstream; the
#    stock repo's model.py already dispatches on JEV_BACKEND=local)
cp /path/to/xiaojev/integrations/jev-ultrafast/local_model.py jev_ultrafast/local_model.py

# 3. point the decision backend at your xiaojev checkout + checkpoint
export JEV_BACKEND=local
export XIAOJEV_HOME=/path/to/xiaojev          # only if auto-detection fails
export XIAOJEV_CKPT=/path/to/xiaojev/ckpt/v3  # default: $XIAOJEV_HOME/ckpt/v3

# 4. TYPE_TEXT (free-form text the decision model intentionally does not
#    produce) is delegated to any local OpenAI-compatible server, e.g. vLLM:
export TEXT_MODEL_BASE_URL=http://127.0.0.1:8020/v1
export TEXT_MODEL=qwen3.8-27b
export TEXT_MODEL_API_KEY=local               # any non-empty string

# 5. run the static fixture (no paid APIs involved)
python scripts/smoke.py
```

## Status: work in progress

On the bundled travel fixture ("find Design stays in Lisbon with Free
cancellation, then open Casa Flora"):

- **v2** checkpoint: 0/2 runs — loops on navigation.
- **v3** checkpoint: qualitative step forward — the very first decision picks
  the Destination input box (probability 1.0) and the text helper correctly
  produces "Lisbon" — but the agent then re-fills the field 4 times instead of
  confirming the autocomplete suggestion, until the no-progress guard blocks
  the run.

Interpretation: operation-level understanding is in place; web-interaction
common sense (autocomplete confirmation, filter toggles) is not — it is a
data-coverage gap, to be closed with P4 browser-domain training data. Details:
[`results/fixture_v3_summary.json`](../../results/fixture_v3_summary.json).
