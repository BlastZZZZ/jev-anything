# Browser adaptation and local regression

Install the [jev-ultrafast integration](../../integrations/jev-ultrafast/README.md) first. Commands below run from the xiaojev repository root, with the model dependencies installed and the fixture server and Chrome already running. The adaptation starts from the original `ckpt/v4` checkpoint.

Set `XIAOJEV_HOME` to this repository and `XIAOJEV_JEV_HOME` to the patched agent checkout. `XIAOJEV_BASE_MODEL` can point to a cached Qwen3-0.6B tokenizer/model. Select your assigned GPU through `CUDA_VISIBLE_DEVICES`; the scripts do not choose a physical GPU. Working data and logs default to `results/v4_browser` (`XIAOJEV_BROWSER_RUN` overrides it). The fixture URL defaults to `http://127.0.0.1:8766/fixture.html?scenario=travel` (`XIAOJEV_FIXTURE_URL` overrides it).

```bash
python experiments/v4_browser/make_browser_mixed.py
python experiments/v4_browser/train_browser.py
python experiments/v4_browser/collect_browser_states.py
python experiments/v4_browser/train_browser.py \
  --data results/v4_browser/browser_dom.jsonl \
  --init ckpt/v4_browser_repair/step300 \
  --steps 150 --eval-every 50 --dev-n 150 --name v4_browser_dom --prefix dom_
```

The first stage uses 16,547 programmatically labeled mixed-control samples. The second stage observes 900 counterfactual pages rendered by the actual fixture and yields 5,007 rows, split into 4,179 training and 828 dev rows with no shared scenario IDs. It also consistently permutes observed element indices. Only the last six backbone layers, normalization layers, and scoring head are updated. Dataset samples are in `data/samples/`.

The trainer saves the best fixed-dev checkpoint. Deployment also requires actual browser regression: the reported second-stage step100 had dev accuracy 82% and passed 4/4 twice; step150 had dev accuracy 86% but passed only 3/4, so it was rejected. Do not automatically deploy the highest-dev checkpoint.

```bash
python experiments/v4_browser/browser_acceptance.py \
  --checkpoint ckpt/v4_browser_dom/step100 --tag final
# After independently verifying the report, create the default alias once:
ln -s v4_browser_dom/step100 ckpt/v4_browser
```

The acceptance script checks the actual detail URL, filter text, and completion state, and saves full decision traces. Its `TEXT_MODEL_*` environment variables configure an OpenAI-compatible text helper; this service supplies only input-field text. The script defaults to a local Qwen server and does not require a paid API.

These cases participate in deployment selection and share the fixture layout with adaptation data. They are regression evidence, not an independent general-web benchmark. The original v4 has previously seen the evaluation entities, even though the adaptation goals exclude them. Preserve failed runs as well as successful ones. Frozen evidence is under `results/v4_repair`, separate from working outputs.
