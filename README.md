# NLP_RAG — Member A Baseline（审计完整交付版）
> **GitHub Repository Note**
>
> 本仓库包含成员 A 的基线源代码、配置、数据、测试和评估结果。
> 完整的历史追踪记录（outputs/a6_3_full/traces.jsonl）因文件体积原因未纳入 Git。
> 如需复现完整的历史审计，需从 GitHub Releases 下载 NLP_RAG_Baseline_Audited_Delivery_Final.zip。
> The historical baseline在原始（raw）口径下达到 45.62% 的准确率，在经表面归一化（surface-normalized）的官方脚本口径下达到 61.35% 的准确率。重构版本尚未在全部 2,556 道问题上重新运行。
## 1. 项目和交付定位

此 ZIP 为 A1–A7 的 **BM25 + Ollama/Qwen3 基础 RAG** 交接物。保留了检索代码、原始知识库及 Chunks、冻结配置、历史 2556 条原始/Surface 预测、历史原始生成 Trace 与 111 条恢复记录，以及四题真实 Pilot 证据。**不是** B 阶段的增强多跳检索系统，也不包含 A6.5 的实验性模型变体。建议下阶段 B 以此固定基线进行对照实验。

- 历史冻结预测（旧 A6.3 + Recovery + A6.4）的官方脚本指标：Raw **45.62%**（1166/2556），Surface **61.35%**（1568/2556）。
- 402 条新增官方命中来自 *gold-free* 的 Yes/No 标点格式规范化；**不是** 检索或语义推理改善。该脚本按空白分词交集判分，不等同严格语义正确率。
- 重构后新版 `src/baseline/` 的完整 2556 题重新生成**未执行**；已有证据为独立目录真实四题 Pilot 及离线完整性验证。
- 官方 65% 为课程最终 Response Accuracy 单项 Pass 档，不是 A 成员独自必须达到的交接门槛，也不是整个小组的最终综合成绩。

## 2. 实际包含内容（以此 ZIP 为准）

```text
NLP_RAG/
  README.md                         # 主入口，当前文件
  config/a6_baseline_frozen.json
  src/baseline/                    # 正式生成及辅助模块，8 个 .py
  src/chunk_corpus.py              # 分块方法
  data/raw/{corpus.json,MultiHopRAG.json}
  data/processed/chunks.jsonl      # 4573 块
  outputs/chunk_validation.json
  outputs/a6_final_baseline/       # 历史 2556 条最终预测与冻结审计摘要
  outputs/a6_3_full/              # 原始 2556 题预测、完整 Trace、恢复 Trace/Manifest
  outputs/a6_4_format_baseline/    # 历史 Surface 和格式报告
  outputs/baseline_runs/pilot_4_seed20261025/  # 新版四题真实 Pilot
  archive/a5_a6_legacy_20261023/frozen_engines/  # 6 份历史校验源码
  eval_tutorial/                   # 官方 prepare_eval.py 与 qa_evaluate.py
  tests/test_baseline_package.py
  docs/{DELIVERY_NOTES.md, A7_B_HANDOFF.md, ...}
  requirements.txt
  SHA256SUMS.txt
```

该交付包**不包含**早期开发工具 `src/download_raw_dataset.py`、`src/inspect_dataset.py`、`tools/migrate_baseline_package.py`、`src_snapshot/` 和未采用的 A6.5 诊断实验；这些只保存在 A 的开发存档。`__pycache__`、`.pyc` 已排除。不要因为旧材料提到上述文件而尝试运行它们。

`archive/.../frozen_engines/` 中 6 份历史源码不是日常调用链，但 `baseline.py` 的配置历史 SHA256 校验需要它们；`outputs/chunk_validation.json` 是 BM25 Chunks 校验依赖。**不要删除这两部分。**

## 3. 从零部署（接收者）

已验证参考平台：Ubuntu/WSL、Python **3.11**、本地 Ollama；本包的 Python runtime 仅明确要求 `rank-bm25==0.2.2` 和 `tqdm==4.67.1`，其他调用主要使用标准库。无需为本 Baseline 单独安装 PyTorch/CUDA。Ollama 是独立服务，**不包含在 pip requirements 里**。

解压到任意工作目录，进入 `NLP_RAG/`，再运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/baseline/baseline.py config
python src/baseline/baseline.py preflight
```

上述操作不会调用 LLM；需联网安装 pip 包。真正生成前安装并启动 Ollama，确认：

```bash
ollama --version
ollama list
curl http://127.0.0.1:11434/api/tags
```

冻结模型标识为 `qwen3:4b-instruct-2507-q4_K_M`，必须保证 `ollama list` 中存在此模型标签并可调用；如果本机不存在，需通过可用的模型仓库/组员提供的模型文件安装。**没有验证任何新机器可直接拉取该特定标签**；请勿无依据地改成别的模型仍沿用历史指标。服务地址配置在 `config/a6_baseline_frozen.json`，迁移至不同 Ollama 位置应另建配置版本并记录修改。

## 4. 主要命令（在 NLP_RAG 根目录运行）

```bash
# 离线数据和配置检查
python src/baseline/baseline.py preflight

# 已附历史完整原始 Trace，可不调用模型复核完整 2556 题历史链路
python src/baseline/baseline.py audit-history

# 5 项离线测试（本包提供所有历史输入，不应再跳过）
python -m unittest discover -s tests -p 'test_baseline_package.py' -v

# 独立复核冻结后的历史官方评分（仅评估，不调用 Ollama）
python src/baseline/baseline.py evaluate \
  --predictions outputs/a6_final_baseline/predictions_surface.json \
  --output-dir outputs/my_eval_01

# 真实四题新实验：调用 Ollama；选择未占用的 seed
python src/baseline/baseline.py pilot --count 4 --seed 20261102

# 针对新 Pilot 运行 Surface；须用实际的 seed 目录与新的目标目录
python src/baseline/baseline.py surface \
  --predictions outputs/baseline_runs/pilot_4_seed20261102/predictions.json \
  --output-dir outputs/baseline_runs/pilot_4_seed20261102/surface_export

# 真正全量重算 2556 题：会调用 Ollama；仅在必要时执行
python src/baseline/baseline.py full --confirm-full-run

# 新版 Full 之后先 dry-run 查看需要恢复的情况
python src/baseline/baseline.py recover
# 确认后才执行（可能调用 Ollama）
python src/baseline/baseline.py recover --confirm-recovery
```

所有 `evaluate` 和 `surface` 的输出目录必须**尚不存在**，避免覆盖已有结果。真实 Full 会写入 `outputs/baseline_runs/full/`，不会覆盖历史 `outputs/a6_3_full/`。原始 Full 的常规恢复机制可能无法处理未保存 `llm_request` 的前置异常，遇到这些异常先保留日志并交开发成员排查，不要手动制造预测数据。

## 5. 可靠性与实验边界

1. A1–A5 基础数据、Chunking、BM25 检索和本地生成工作已实现。实际单轮 BM25 Top-K=10 后再按文档去重，若重复较多不会自动补足十篇；Context 默认最多 8500 字符，遇到首个放不下的 Chunk 会停止添加后续候选。
2. Prompt 要求简洁、仅根据证据回答，但无法保证多跳证据充分性、比较和时间顺序推理正确；`success` / `stop` 只表示完成执行，不是语义正确。
3. `done_reason == length` 归为不完整，其余未知结束原因的状态处理有局限；Recovery 对请求尚未生成的错误也有边界。未对历史冻结核心进行隐藏修改。
4. 禁止在正式生成、检索或输出规范化中使用 `answer`、`evidence_list` 或针对特定 QID 的标准答案规则。官方评分器读取 Gold 是允许且必须隔离的环节。
5. 预处理 Chunk 长度为 300 英文词、重叠 50 词；标题/元数据带入 Evidence，但 BM25 索引使用 `text` 字段。
6. 实验文件的 Manifest、Trace 与 SHA256 是工程溯源材料，不能证明模型真实多跳推理能力。

## 6. A→B 接手及团队提交

详细接口和待办见 `docs/A7_B_HANDOFF.md`。B 的任务是多跳检索、检索增强、推理/证据整合及消融实验；A 的 Frozen Baseline 应保留不变作为对照。C 负责正式 Benchmark、错误分析和最终报告。

**GitHub 仓库 URL/权限不在 ZIP 中，必须由项目维护者单独提供给 B**。本包可离线传递源码，但单纯交一个 ZIP 不等于完成“源码放在 GitHub”的课程提交要求。完整 Trace `traces.jsonl` 可能超过 GitHub 单文件上传限制；不应强行直接用普通 Git 推送超大文件，可通过项目允许的 Git LFS / Release / 外部文件链接交付，并同时保存 SHA256 校验。

首次接手应执行 `preflight → unit tests → audit-history → evaluate`，之后再配置 Ollama 做小样本 Pilot。历史全量分数属原 A6.3 冻结记录，不能包装为新版 Full 得分。
