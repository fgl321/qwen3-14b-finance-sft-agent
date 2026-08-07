# RAG 与端到端评测

## 怎么跑

1. 启动完整服务（Agent API :8002，PostgreSQL/Redis/Qdrant 可用，模型服务可用）。
2. 入库评测文档并运行用例：

```bash
cd agent
python -m scripts.run_production_eval \
  --base-url http://127.0.0.1:8002 \
  --ingest-dir ../docs/eval/documents \
  --cases ../docs/eval/cases.jsonl \
  --output-dir reports/production_eval \
  --user-id eval_user
```

脚本会先把 `docs/eval/documents/` 下的评测文档写入知识库（内容哈希去重），
再逐个调用生产接口 `/api/chat/graph-v2`，最后输出 `production_eval.json`
和 `production_eval.md` 报告。

## 用例覆盖

`cases.jsonl` 共 13 个用例，覆盖：

- 概念问答（带知识库引用）：市盈率、夏普比率
- RAG 命中：紧急备用金、寿险缺口、负债管理
- RAG 无证据拒答：知识库没有的内容不得编造
- 提示注入隔离：文档中的指令只能当数据，不得执行、不得复述攻击短语
- 确定性数值复算：18 万 ÷ 12 = 1.5 万、备用金 4.5 万 ~ 9 万
- 短期记忆（Redis）：多轮对话内回忆用户上轮提供的信息
- 长期记忆（PostgreSQL）：跨线程回忆用户长期事实
- 风险边界：不承诺收益、不推荐具体证券
- 通用金融概念

## 判定方式

- 客观指标确定性计算：Recall@3/Recall@5、MRR、nDCG@5（按期望命中的文件名）、
  引用命中率、引用精度、禁止词安全兜底。
- 语义判定采用 LLM-as-a-judge：裁判模型按固定 JSON
  （`verdict` / `score` / `reason` / `issues`）评判回答是否正确、
  是否按要求引用/拒答、是否安全。裁判调用失败时回退到确定性判定。
