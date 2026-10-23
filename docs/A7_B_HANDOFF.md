# 成员 A → 成员 B 交接说明

## A 已完成范围

- A1 环境选型及目录（Python 3.11 / Ubuntu WSL / Ollama 的参考配置）；A2 官方 Corpus 609 篇和 2556 问题核验；A3 300-word / 50 overlap Chunks，共 4573；A4 `BM25Retriever`；A5 基础 RAG 及实际 Evidence + Prompt + Ollama；A6 历史全量运行、恢复与 Surface 研究及 Pilot；A7 交付代码与历史可审计证据。
- A 的冻结历史 Raw 45.62%、Surface 61.35%；这是官方空白词交集匹配率，**不是语义准确率提升 15.73 百分点**。
- 新的 `src/baseline/` 只做过四题实际 Ollama Pilot，尚无新目录 2556 题重算成绩。历史详细 Trace 可在 `outputs/a6_3_full/traces.jsonl` 查看。

## 对接接口

- `data/processed/chunks.jsonl`：每块含 `chunk_id, doc_id, text` 和标题/来源/时间/原文位置等信息，具体以文件实际字段为准。
- `src/baseline/bm25_retriever.py`：`BM25Retriever(chunks).retrieve(question: str, top_k: int)`；返回按分数排序的 `dict` 列表，每条包含 `rank`, `score` 和原始 Chunk 字段。
- `src/baseline/rag_pipeline.py`：`first_chunk_per_document()`, `make_context()`, `make_user_prompt()`, `run_one()`；`run_one()` 产出含检索排名、实际 Context、LLM Request、模型回答和状态的 Trace。
- `src/baseline/batch_runner.py`：按 `question_id` 的 Manifest + Checkpoint 批处理机制，保留初始 A Baseline 的输出目录、源码与数据哈希。
- `config/a6_baseline_frozen.json`：冻结 BM25 Top-K=10、unique_docs=true、max_context_chars=8500、Qwen3/Ollama 参数。不要在历史 Baseline 下静默改写。

## B 阶段建议优先实验

1. **证据召回**：观察每题是否检索到了所需的第一跳、第二跳来源；扩展候选集合、单文档去重顺序与 Context Budget（现有先 top-K 后去重，可能只剩 3–4 个 Evidence）。
2. **增强检索**：Dense/Hybrid、Rerank、多跳或迭代查询；必须报告与冻结 A 的对照及误差来源，不能保证自动提分。
3. **生成与验证**：针对 comparison/temporal 的时间和实体约束，处理过度拒答和 Yes/No 与最终回答矛盾；不能从 Gold 构造回答规则。
4. **接口独立性**：当前 `rag_pipeline.py` 与具体 BM25 类型直接耦合；B 要替换检索器时可以在新分支建立统一 Retriever 协议，而非强改 A 基线源码。

## 已知工程问题（不构成 B 前的阻塞）

- Recovery 对错误发生在 `llm_request` 保存之前的 Trace 没有通用恢复路径；Batch Checkpoint 对已有失败状态直接跳过；下一轮正式 Full 前应增补行为级单测。
- 某些 `done_reason` 可能未被严格视为非成功；应在新版本独立实验中校验终止状态。
- 历史官方得分受输出标点规则强烈影响；C 的正式报告应并列展示 Raw / Surface，并另外分析检索证据充分性与语义准确性。

## 交接前人工事项

- 项目维护者提供 GitHub 仓库 URL、访问权限和主分支；本 ZIP 并不证明仓库已配置。
- B 解压后先运行 Preflight、五项离线单测、完整历史 Audit、冻结预测的官方 Evaluation，确认工具链完整，再运行自己的检索改进实验。
- 不要覆盖旧目录 `outputs/a6_3_full/`、`outputs/a6_4_format_baseline/`、`outputs/a6_final_baseline/`，新增实验采用独立路径和 Manifest。
