# Qwen3-14B Finance SFT Agent

这是一个完全自托管的中文金融 Agent。下载者在自己的 GPU 上运行模型；项目作者不提供长期在线推理服务。

最终模型由以下四部分共同组成，缺少任意一项都不是本项目的最终 SFT 模型：

1. 官方基座 `Qwen/Qwen3-14B`
2. 本项目 GitHub Release 中的最终 SFT LoRA
3. Release 中的扩充 tokenizer
4. Release 中的 `embedding_patch.pt`

Agent 使用 DeepSeek V4 Flash 负责意图路由、规划、计划复核和输出防护；最终回答可由用户在每次请求中选择本地蒸馏 Qwen3-14B Finance SFT 或 DeepSeek API。金融计算由十个确定性白名单工具完成，不让语言模型心算关键数字。

系统采用显式 LangGraph 状态机、混合检索 RAG、短/长期记忆、异步文档入库、安全 SSE 进度和可审计工具轨迹。完整设计见 [系统架构](docs/ARCHITECTURE.md)。

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

未构建前端时会打开内置后备页。正式展示请构建 React 前端，构建产物会被 Agent 自动托管：

```powershell
cd frontend
pnpm install
pnpm build
```

然后重启 Agent 服务即可。

前端支持：

- **最终回答模型切换**：每个请求可选择「蒸馏 Qwen3-14B」（本地 SFT 模型）或
  「DeepSeek API」，无需重启服务，切换通过 `synthesis_llm_provider` 字段下发。
- **文档范围**：上传后自动切换到「该文档」范围（也可手动选择「全部文档」），
  检索会限定在所选文档内；对「这个文档讲了什么」类问题会自动回退到文档位置序，
  保证回答始终基于刚上传的文档。
- **异步入库**：PDF、DOCX、TXT、MD 上传后立即返回任务编号；解析、切块和向量化不阻塞 API，页面持续显示 queued/processing/completed/failed 状态。
- **可控执行**：安全 SSE 展示当前节点；用户可停止生成。完成后可展开查看节点轨迹、工具名、状态和耗时，但不会暴露模型思维链。
- **知识与记忆**：知识库列表、按文档检索/删除、会话隔离和长期记忆清理。

## 架构

外层是确定性的状态机，内层只允许模型在有预算的节点中做有限自主决策：

```text
request boundary -> intent router -> planner (最多 3 轮) -> plan review
  -> tool executor (0..N, bounded) -> observation validator
  -> result validator -> answer synthesis (Qwen | DeepSeek)
  -> output guard -> trace finalizer
```

RAG 在请求边界编排：BGE-M3 dense+sparse 混合召回、父子块映射、BGE 重排、证据充分性和引用生成。PostgreSQL 保存 LangGraph checkpoint/长期事实，Redis 保存短期记忆，Qdrant 保存知识向量。

## 目录

```text
agent/               FastAPI、LangGraph、RAG、记忆、工具和测试
frontend/            React 对话与知识库界面
model/               最终 SFT 模型加载器与本地 OpenAI 兼容服务
scripts/             下载、启动脚本
docs/                模型与交付说明
```

## 安全与限制

- 不要提交 `.env`、API Key、SSH Key 或用户金融数据。
- 这是固定 `personal/owner` 身份的单用户部署；浏览器提交的 tenant/user 不作为授权依据。若开放公网或改成多用户，必须增加真实认证、RBAC、限流和租户隔离。
- 上传执行扩展名、MIME、文件签名和大小检查，使用随机存储文件名；工具注册表启动后冻结，所有循环和工具调用都有硬预算。
- 本项目用于金融信息整理与辅助分析，不构成投资、法律、税务或持牌金融建议。
- 模型可能产生错误数字、过时结论或不完整风险提示；生产使用前必须建立业务侧审核、监控和回滚机制。
- 本仓库代码与最终 SFT 增量包采用 Apache License 2.0；官方 Qwen3-14B 基座不包含在本仓库中。

## 质量门槛

提交级 CI 统一运行全部后端测试、Python 编译检查、密钥/权重误提交检查和前端生产构建。发布级评测使用固定 1500 条金融盲测，在同一配置下比较基座、蒸馏和 SFT 模型，并分别测试 Qwen/DeepSeek 最终回答。指标、数据隔离和发布阻断条件见 [评测与发布门槛](docs/EVALUATION.md)。

当前工作区联合验收：后端 `317 passed`，前端 TypeScript 检查与 Vite production build 通过（2026-08-13）。

历史模型包验证：

- 模型增量包：`qwen3-14b-finance-sft-adapter-v1.tar.gz`
- 压缩包大小：242,086,199 bytes
- SHA-256：`4447f31637905b5a51aaf6a99bf2c1397c21e0dbebd0d44fcb5981fca8af739d`
- 独立自托管加载：通过（NVIDIA A100 80GB，Qwen3-14B 基座 + Release 包）
- RAG/端到端历史评测（13 个用例）：13 passed；
  Recall@3/Recall@5 = 1.0、MRR = 1.0、nDCG@5 = 1.0、引用命中率 = 1.0。
  覆盖概念问答、RAG 命中/无证据拒答、提示注入隔离、数值复算、
  短期记忆、长期记忆与风险边界。详见 `docs/eval/`。
- 公开金融文档演示：`docs/eval/documents/基金投资新手攻略.md`（内容整理自
  中国证监会陕西监管局投资者教育栏目《权益360：基金投资新手攻略》），
  配套测试题见 `docs/eval/finance-doc-test-questions.md`。

模型包的历史结果见 `docs/RELEASE_VALIDATION.md`。历史报告不自动证明当前代码版本通过；以当前提交的 CI 和重新生成的评测报告为准。
