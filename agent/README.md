# Finance Agent

这是最终 SFT 模型的 Agent 应用层。模型必须由下载者在自己的 GPU 上启动，默认 OpenAI 兼容地址为 `http://127.0.0.1:8001/v1`。

- Qwen3-14B + 最终 SFT LoRA/tokenizer/embedding patch：只负责最终答案合成。
- DeepSeek API：负责路由、任务规划、工具编排、复核、RAG 和记忆抽取。
- PostgreSQL：长期事实。
- Redis：短期记忆。
- Qdrant + BGE-M3：知识检索。

完整安装与模型下载说明见仓库根目录 `README.md`。

## 配置

```powershell
Copy-Item .env.example .env
```

至少填写：

- `DEEPSEEK_API_KEY`：下载者自己的 DeepSeek Key。
- `QWEN_API_KEY`：本地模型服务的 Key。
- `QWEN_BASE_URL=http://127.0.0.1:8001/v1`。

真实 `.env` 不得提交到 GitHub。

## 启动

先在仓库根目录启动模型服务，然后在本目录运行：

```powershell
docker compose up -d
python -m scripts.init_personal_data
python -m scripts.run_production_api
```

不要使用 `--reload`。Windows 下项目使用自定义 SelectorEventLoop 兼容 PostgreSQL 异步连接。

## 入口

- 网页：`http://127.0.0.1:8002/`
- 健康检查：`GET /health`
- Qwen 健康检查：`GET /health/qwen`
- DeepSeek 健康检查：`GET /health/deepseek`
- 生产聊天：`POST /api/chat/graph-v2`
- OpenAPI：`http://127.0.0.1:8002/docs`

## 验收

```powershell
python -m pytest -q
python -m scripts.run_all_checks
```
