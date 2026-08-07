# Qwen3-14B Finance SFT Agent

这是一个完全自托管的中文金融 Agent。下载者在自己的 GPU 上运行模型；项目作者不提供长期在线推理服务。

最终模型由以下四部分共同组成，缺少任意一项都不是本项目的最终 SFT 模型：

1. 官方基座 `Qwen/Qwen3-14B`
2. 本项目 GitHub Release 中的最终 SFT LoRA
3. Release 中的扩充 tokenizer
4. Release 中的 `embedding_patch.pt`

Agent 使用 Qwen3-14B Finance SFT 负责最终回答，使用下载者自己的 DeepSeek API Key 负责路由、规划、复核、工具编排、RAG 和记忆抽取。

## 硬件与空间

- 推荐：单张 40GB 或更大显存的 NVIDIA GPU；A100 80GB 已实测通过。
- 模型精度：BF16。
- 基座下载约 28GB，另需模型缓存、BGE-M3 和运行空间。
- 24GB 显卡不能直接按本仓库默认 BF16 配置稳定加载；自行量化会改变本项目交付精度。

## 快速开始

### 1. 克隆与安装

```powershell
git clone https://github.com/fgl321/qwen3-14b-finance-sft-agent.git
cd qwen3-14b-finance-sft-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r model\requirements.txt
python -m pip install -r agent\requirements.txt
```

Linux 将虚拟环境激活命令替换为 `source .venv/bin/activate`。

请根据自己的 CUDA 版本优先按 PyTorch 官方说明安装 GPU 版 PyTorch，再安装其余依赖。

### 2. 下载最终 SFT 增量包

发布后执行：

```powershell
.\scripts\download_model.ps1 -Repo "fgl321/qwen3-14b-finance-sft-agent"
```

Linux：

```bash
bash scripts/download_model.sh fgl321/qwen3-14b-finance-sft-agent
```

脚本只下载本项目的 SFT LoRA、扩充 tokenizer 和 embedding patch。官方 Qwen3-14B 基座会在首次启动模型服务时由 Transformers 自动下载，也可通过 `QWEN_BASE_MODEL` 指向提前下载的本地目录。

### 3. 启动本地模型服务

```powershell
$env:QWEN_SERVER_API_KEY="replace-with-a-local-secret"
.\scripts\start_model.ps1
```

默认地址为 `http://127.0.0.1:8001`。启动时会严格检查 adapter、tokenizer 和 embedding patch。

### 4. 配置并启动 Agent

```powershell
Copy-Item agent\.env.example agent\.env
```

编辑 `agent/.env`：

- 填写下载者自己的 `DEEPSEEK_API_KEY`。
- 将 `QWEN_API_KEY` 设置为第 3 步的本地密钥。
- 保持 `QWEN_BASE_URL=http://127.0.0.1:8001/v1`。
- 默认 `SYNTHESIS_LLM_PROVIDER=qwen`（最终回答由本地蒸馏模型生成）。
  如需先用 DeepSeek 验证全链路，可临时改为 `deepseek`。

随后执行：

```powershell
.\scripts\start_agent.ps1
```

浏览器访问 `http://127.0.0.1:8002/`。

## 架构

```text
Browser :8002
    -> FastAPI / LangGraph Agent
       -> DeepSeek API：路由、规划、复核、RAG/记忆编排
       -> PostgreSQL：长期事实
       -> Redis：短期记忆
       -> Qdrant + BGE-M3：知识检索
       -> Local Qwen :8001：最终答案生成
          -> Qwen3-14B base
          -> final SFT LoRA
          -> expanded tokenizer
          -> embedding patch
```

## 目录

```text
agent/               金融 Agent、前端、Docker 依赖和测试
model/               最终 SFT 模型加载器与本地 OpenAI 兼容服务
scripts/             下载、启动脚本
docs/                模型与交付说明
```

## 安全与限制

- 不要提交 `.env`、API Key、SSH Key 或用户金融数据。
- 本项目用于金融信息整理与辅助分析，不构成投资、法律、税务或持牌金融建议。
- 模型可能产生错误数字、过时结论或不完整风险提示；生产使用前必须建立业务侧审核、监控和回滚机制。
- 本仓库代码与最终 SFT 增量包采用 Apache License 2.0；官方 Qwen3-14B 基座不包含在本仓库中。

## 验证状态

- 模型增量包：`qwen3-14b-finance-sft-adapter-v1.tar.gz`
- 压缩包大小：242,086,199 bytes
- SHA-256：`4447f31637905b5a51aaf6a99bf2c1397c21e0dbebd0d44fcb5981fca8af739d`
- 独立自托管加载：通过（NVIDIA A100 80GB，Qwen3-14B 基座 + Release 包）
- Agent 单元测试：188 passed、1 skipped
- RAG/端到端评测（13 个用例）：12 passed、1 known issue（记忆确认时回答略冗余）；
  Recall@3/Recall@5 = 1.0、MRR = 1.0、nDCG@5 = 1.0、引用命中率 = 1.0。
  覆盖概念问答、RAG 命中/无证据拒答、提示注入隔离、数值复算、
  短期记忆、长期记忆与风险边界。详见 `docs/eval/`。

详细结果见 `docs/RELEASE_VALIDATION.md`。
