# xiaojev（小 Jev）

**开源、校准的 System One 式决策模型。** 输入 state + 动态候选集，一次前向直接输出候选上的完整概率分布——只做选择，不生成文本。可以理解为一个小型、开源、可本地运行的 [Choice 原语](https://docs.typesafe.ai/primitives/choice) 替代品，用程序真值监督而非教师模仿来获得校准。

[English README](README.md) · [完整结果](docs/RESULTS.md)

## 核心结果速览

| | **xiaojev v2** (0.6B) | Jev（闭源 API） | NanoJev（开源） | 零样本 Qwen3-0.6B |
|---|---|---|---|---|
| 概率域 test TV ↓ | **0.099** | 失校准（均匀硬币→0.89–0.95） | 0.463 | 0.471 |
| 概率域 test acc ↑ | **0.881** | — | 0.385 | 0.497 |
| 跨原语一致性 ↓ | **0.027** | Choice/Noul 差 ~0.4 | 0.374 | 0.754 |
| 游戏 macro（test / ood） | **48.8% / 28.7%** | 65.4% / 43.7% | 66.8% / 45.5% | 15.4% / 12.2% |
| 游戏上对 Jev（配对 McNemar） | 八格全部 **p > 0.16** | — | — | — |
| 单决策延迟 | **46–85 ms**（单卡 3090） | 网络 API | 57–84 ms（同 GPU） | — |
| 单决策成本 | **$0（本地）** | 付费 API | $0（本地） | $0（本地） |

一个模型同时做到：概率域校准远超 Jev，游戏上与 Jev 无显著差异。全部细节见 [docs/RESULTS.md](docs/RESULTS.md)。

## 五个发现

**1. 知行差距。** 通用 LLM（Qwen3.8-27B）用自然语言报概率时近乎完美校准（mean |err| 0.005，含 0.63×0.41=**0.2583** 四位精确），但经 token 选择通道表达同一信念时严重失真：均匀硬币 0.953/0.047（TV 0.45）、K=20 均匀抽签 TV 0.55、位置/标签偏置、跨原语矛盾（同一 state 下 Choice 说正面 0.95 而 Noul 说 P(正面)=0.053）。五种替代诱导方法均不能修复——只有语言通道是校准的。

**2. 闭源 Jev API 概率也不准。** 明示均匀硬币下，Jev Choice 报 P(正面) **0.89–0.95**，Noul 对同一事件报 0.48–0.49（差 ~0.4），而让它直接识别先验时 4/4 答对。K=255 均匀抽签每项报 0.75/0.84（参考 1/255）。（数据：NanoJev 公开概率语义探针 + 我们的复核回执；我们的 27B 探针呈现同一模式。）

**3. 程序真值监督可搬运校准。** 9 万道程序生成概率题（精确解析解、无 LLM 参与）+ 分布监督（软标签 CE + Brier），把 Qwen3-0.6B 的 test TV 从 **0.471 压到 0.082**，跨原语一致性 0.754 → **0.024**——比 Jev 更准。未见机制（扑克、骰子和）OOD Brier **0.019–0.108**：真泛化。

**4. 专精不泛化。** 与 NanoJev 双向交叉对照：我们的概率专精 v1 在游戏上只有 **11.2%**（低于共享基座 15.4%）；NanoJev 公开权重在我们的概率测试上只有 **38.5%**（同样低于基座 49.7%）。打分头架构没有魔法，泛化来自数据覆盖。

**5. 混合域训练 = 通用 System One。** v2（同一 0.6B 架构，概率:游戏 1:1 混合、2000 步、从头训练）两域同超基座：概率 acc **0.881** vs 0.497，游戏 macro **48.8%** vs 15.4%；游戏上与 Jev 八格无显著差异（McNemar p>0.16），Maze 6/10 赢 NanoJev 4/10；单决策 **46–85ms**（单卡 3090），概率域校准远超 Jev。

## 架构

```
 state + question + K 个候选（动态集合）
          │
          ▼
 chat 模板（单条 user 消息）
          │
          ▼
 K 条候选路径：[prompt][label k][EOS]    label：0-9、A-Z（单 token）
          │
          ▼
 Qwen3-0.6B 主干 —— 一次打包前向（bf16 autocast，fp32 权重）
          │
          ▼
 每条路径 EOS 位置隐状态
          │
          ▼
 LayerNorm → Linear（每路径一个标量分）
          │
          ▼
 组内 softmax ——► 候选上的完整概率分布（不生成文本）
```

训练损失：组内 softmax 之后对精确目标分布的软标签交叉熵 + 0.1 × Brier。

## Quickstart

```bash
pip install -r requirements.txt
```

### 复现数据

```bash
# 概率题：9 万行、精确解析目标、seed 固定（20260921）
python data/datagen.py --n 90000 --out data/train_v1.jsonl
pytest data/test_datagen.py        # 11 个自洽性测试（暴力枚举 oracle）

# 游戏决策：从 NanoJev-Data 转换（Jev 教师 / 视觉专家软标签）
huggingface-cli download C-Tianyu/NanoJev-Data --repo-type dataset --local-dir /path/to/NanoJev-Data
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data
python data/make_gamedata.py       # -> data/games_v1.jsonl
```

两个数据集的各 100 行样本已提交在 `data/samples/`，不下载也能查看格式。

### 训练

```bash
# v1 —— 概率专精（1200 步，单卡 3090 约 6 小时）
python training/train.py --steps 1200 --out ckpt/v1

# v2 —— 通用 System One：概率:游戏 1:1（2000 步，约 10 小时）
python training/train.py --steps 2000 --data data/train_v1.jsonl \
    --mix data/games_v1.jsonl --mix-ratio 0.5 \
    --out ckpt/v2 --log results/train_log_v2.jsonl
```

主干默认 `Qwen/Qwen3-0.6B`（可用 `XIAOJEV_BASE_MODEL` 覆盖）。

### 评测

```bash
python training/evaluate.py --ckpt ckpt/v2 --split test --output results/eval_v2_test.json
python training/evaluate.py --ckpt ckpt/v2 --split ood  --output results/eval_v2_ood.json
python training/evaluate.py --zero-shot --split test --output results/eval_zeroshot_test.json
```

指标：acc / NLL / Brier / ECE(10) / 平均 TV（总体 + 按机制×难度）+ 配对 Choice-vs-Noul 一致性。

### 复现零样本探针（发现 1）

需要本地 vLLM 服务；我们用的是 `cyankiwi/Qwen3.8-27B-AWQ-INT4`：

```bash
vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 --port 8020 --served-model-name qwen3.8-27b
export XIAOJEV_VLLM_URL=http://127.0.0.1:8020/v1   # 默认值；XIAOJEV_VLLM_MODEL/XIAOJEV_MODEL_PATH 同理

python probes/probe_distributions.py   # token 通道失真、跨原语矛盾
python probes/probe_verbalized.py      # 语言通道校准
python probes/probe_novel.py           # 全新（不可记忆）机制
python probes/probe_elicitation.py     # 五种诱导方法对比
```

### 复现对比实验（发现 4–5）

```bash
export XIAOJEV_NANOJEV_REPO=/path/to/NanoJev        # github.com/TianyuCodings/NanoJev 的克隆
export XIAOJEV_NANOJEV_DATA=/path/to/NanoJev-Data

python comparison/compare_sanity.py      # 复核 NanoJev 已发表的 548 例数字
python comparison/compare_rollout.py --engine vcdm --checkpoint ckpt/v2 \
    --output results/compare_v2_frozen_episodes.jsonl
python comparison/compare_report.py results/compare_v2_frozen_episodes.jsonl \
    results/compare_v2_frozen.json
python comparison/compare_reverse.py     # NanoJev 权重跑我们的概率测试
python comparison/latency_bench.py       # v1 / v2 / NanoJev 延迟（--skip-nanojev 可跳过）
```

（`vcdm` 是 xiaojev checkpoint 的内部代号，为兼容产物文件而保留。Doom 场景的游戏 rollout 另需 `vizdoom`。）

## 完整结果

上文引用的每个数字（含分格表与对应的产物文件）见 **[docs/RESULTS.md](docs/RESULTS.md)**。汇总 JSON 在 `results/`；逐行预测与轨迹可用脚本复现，但未提交（体积）。

## Checkpoints

两个 checkpoint（共约 14 GB，含优化器状态）：

- `xiaojev-v1` —— 概率专精（TV 0.082）
- `xiaojev-v2` —— 通用 System One（概率 + 游戏）

> **TODO：** 将上传到 HuggingFace —— 占位链接 `https://huggingface.co/TODO/xiaojev`。
> 在此之前可用上面的命令从头复现；数据生成 seed 固定，训练 seed 默认 0。

## 环境要求

- 单卡 24GB GPU（开发用一张 RTX 3090）
- Python 3.12+（开发环境为 3.10 + torch 2.10.0+cu128 + transformers 4.57.6）
- `torch` 2.10、`transformers` 4.57、`httpx`、`pytest`；Doom 游戏 rollout 另需 `vizdoom`；27B 探针目标需 vLLM 部署

## 限制

- **研究原型**：两个 checkpoint、一张卡、一个随机种子，未做大规模超参搜索。
- **8K 上下文**：超出 8192 token 预算的 state 会被截断。
- **域覆盖**：只训练了程序概率机制与四个游戏任务（Maze、Snake、Doom basic、Doom predict_position）；浏览器、RAG 等语义决策域未训练未评测。
- **游戏监督是教师蒸馏**（Jev native_probs / 视觉专家策略），不是真值；只有概率域有精确解析目标。
- **与 TypeSafe 无任何关联**：Jev 仅作为基准出现（NanoJev 公开产物与公开 API 回执），xiaojev 是独立的复现研究。

## 致谢

- [NanoJev](https://github.com/TianyuCodings/NanoJev)（MIT）——游戏训练数据（NanoJev-Data）、548 例冻结对比协议与验证器、Jev API 回执。
- [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)（MIT）——browser-use 域的 System One 式决策模型参照。
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)（Apache 2.0）——基座（0.6B）；27B 探针目标为 Qwen3.8-27B 的社区 AWQ 量化。

## License

MIT —— 见 [LICENSE](LICENSE)。衍生自 NanoJev 与 Qwen3 的数据和权重仍遵循各自许可证。
