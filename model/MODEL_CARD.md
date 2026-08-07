# Qwen3-14B Finance SFT

## 模型组成

- 基座：`Qwen/Qwen3-14B`
- 增量训练：BF16 LoRA SFT
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`
- 目标层：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`
- 额外制品：扩充 tokenizer 与 `embedding_patch.pt`

当前最终包中的 `embedding_patch.pt` 经审计存在且 `token_ids` 为空，表示该次 tokenizer 对齐不需要覆盖额外 embedding 行；加载器会验证文件存在后安全跳过空补丁。

只加载官方基座不是本模型；必须通过 `model/model_loader.py` 同时加载 Release 中的最终 adapter、tokenizer 和 embedding patch。

## 最终 SFT 运行记录

- 状态：passed
- 样本数：4,500
- Epoch：1
- 学习率：`5e-6`
- 精度：BF16
- 训练 GPU 进程数：3
- 训练耗时：944.5335 秒
- 最终训练损失：0.9945891380310059
- 数据集 SHA-256：`5e81ca6464f4a6a5df1f849ecc5cade5ecc0e58dce151edafe7b7f7a26725b13`

## 用途

用于中文个人金融问答、现金流和债务分析、常见金融概念解释、金融 Agent 最终答案合成。它不替代持牌投资顾问、律师、会计师或税务专业人员。

## 局限

- 不能保证实时行情、法规或产品信息最新。
- 复杂数字问题仍需确定性工具复算。
- 可能出现幻觉、遗漏风险条件或不适当的确定性表述。
- 生产环境必须配置输入输出审核、日志脱敏、限流、监控和人工升级路径。

## 发布内容

GitHub Release 增量包应至少包含：

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `added_tokens.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `embedding_patch.pt`
- `sft_metadata.json`
- `SHA256SUMS`

## 独立加载验收

- 状态：passed
- 硬件：NVIDIA A100 80GB PCIe
- tokenizer size：151,669
- 加载方式：官方 Qwen3-14B 基座 + GitHub Release 最终增量包
- 实测回答：`市盈率是股票价格与每股收益的比率，用于衡量股票的估值水平。`
