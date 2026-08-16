# 02 Constraint / Strategy 裁决矩阵

> 状态：冻结候选设计的规范性裁决表  
> 文档版本：`constraint-strategy-matrix-v1`  
> 依赖：`01_control_plane_schema.md`  
> 目标：开发人员只需翻译矩阵，不得在代码阶段重新决定产品语义

## 1. 裁决顺序

所有 Strategy Reconciliation MUST 按固定顺序执行：

```text
Authority / Enforcement
→ Permission
→ Scope
→ Requirement
→ Availability
→ Tool Effects
→ Budget / Deadline
→ Strategy
```

后续阶段不得覆盖前一阶段的禁止结果。Reconciler 只能补齐或删除 Strategy Step，不能修改 Sealed Contract。

每次裁决必须产生：

```text
decision
task_impact
health_impact
delivery_impact（若此时可确定）
reason_code
retryable
```

无法匹配任何组合时 MUST fail closed：

```text
decision = do_not_execute_high_risk_capability
reason_code = STRATEGY_UNCLASSIFIED_CONTROL_STATE
```

## 2. Authority 与 Enforcement 裁决

### 2.1 强度顺序

以下顺序只用于不同来源的冲突；同一 Authority 内 `required + forbidden` 进入显式 Conflict：

```text
system_policy + hard_policy
> api_policy + hard_policy
> user_explicit + explicit_constraint
> caller_explicit + explicit_constraint
> semantic_extractor + inferred
> api/ui default
> router_strategy
```

来源通道本身不等于权威。例如 API/UI Default 不能覆盖 User Explicit。

### 2.2 来源冲突矩阵

| 高优先级约束 | 低优先级约束 | 裁决 | Task 影响 | Health 影响 | Reason Code | Retryable |
|---|---|---|---|---|---|---|
| `forbidden` hard policy | `required` | 禁止执行 | `blocked` | `healthy` | `CONTRACT_BLOCKED_BY_POLICY` | false |
| `required` hard policy | `forbidden` 低优先级 | 保留 required；低优先级约束无效 | 继续 | `healthy` | `CONTRACT_INTEGRITY_REPAIRED` | false |
| user explicit `required` | UI default `optional/not_needed` | required | 继续 | `healthy` | `CONTRACT_INTEGRITY_REPAIRED` | false |
| user explicit `forbidden` | Router proposes capability | 删除 Strategy Step | 继续或 blocked（取决于其他 required） | `healthy` | `CONTRACT_PERMISSION_CONFLICT` | false |
| Extractor `required` | Router 未规划 | Reconciler 补齐 | 继续 | Router 可标 degraded | `STRATEGY_RECONCILED` | false |
| Router proposes optional | Contract 没有要求且允许 | 可保留，但受 Budget/Effects 限制 | 不改变 required fulfillment | `healthy` | 无 | false |

### 2.3 同一 Authority 冲突

| 同 Authority 输入 | 是否用户可解决 | 裁决 | Task | Reason Code |
|---|---:|---|---|---|
| 同一 Capability `required + forbidden` | 是 | Seal 前要求澄清 | `blocked` | `CONTRACT_PERMISSION_CONFLICT` |
| 系统策略内部 `required + forbidden` | 否 | 配置错误，禁止执行 | `blocked` | `CONTRACT_BLOCKED_BY_POLICY` |
| “必须用上传文档 + 禁止 Web” | 不构成冲突 | RAG required、Web forbidden、Scope=uploaded | 继续 | 无 |
| “必须用工具 + 禁止外部工具”且存在本地工具 | 不构成冲突 | 只允许本地 Effects 合规工具 | 继续 | 无 |
| “必须用工具 + 禁止任何工具” | 是 | 澄清 | `blocked` | `CONTRACT_PERMISSION_CONFLICT` |

## 3. Requirement / Permission 主矩阵

| Requirement | Permission | Scope | Availability | 基础裁决 | Strategy | Reason Code |
|---|---|---|---|---|---|---|
| required | allowed | valid | available | 必须执行 | 必须包含 Capability Step | 无 |
| required | allowed | valid | degraded | 按 Satisfaction Policy 判断可接受降级 | 包含降级 Step 或标记 unfulfillable | `CAPABILITY_DEGRADED` |
| required | allowed | valid | unavailable | 要求保持 required，但无法执行 | 不创建虚假 Step；记录 unmet | `CAPABILITY_UNAVAILABLE` |
| required | allowed | valid | unknown | fail closed；可做低风险 Availability Probe | 高风险 Step 不执行 | `CAPABILITY_UNAVAILABLE` |
| required | allowed | invalid | 任意 | 不得扩大或替换 Scope | blocked/unfulfillable | `SCOPE_RESOLUTION_FAILED` |
| required | forbidden | 任意 | 任意 | 禁止执行 | blocked/conflict | `CONTRACT_PERMISSION_CONFLICT` 或 `CONTRACT_BLOCKED_BY_POLICY` |
| optional | allowed | valid | available | 可执行 | 仅在 Soft Budget 内加入 | 无 |
| optional | allowed | valid | degraded | 仅当降级仍有价值且预算允许 | 可选降级 Step | `CAPABILITY_DEGRADED` |
| optional | allowed | 任意 | unavailable/unknown | 跳过 | 不加入 Step | `CAPABILITY_UNAVAILABLE`（仅审计） |
| optional | forbidden | 任意 | 任意 | 禁止执行 | 删除 Step | `CONTRACT_PERMISSION_CONFLICT`（若 Router 违规） |
| not_needed | allowed | valid | available | 默认不执行 | Router 可提议低风险补充，但不得影响完成度 | 无 |
| not_needed | forbidden | 任意 | 任意 | 禁止执行 | 删除 Step | `CONTRACT_PERMISSION_CONFLICT`（若 Router 违规） |

本表中的 `blocked/unfulfillable` 是 Strategy 阶段结果，不直接覆盖最终 Task Status。最终仅在任何业务执行尚未合法开始时推导为 `blocked`；如果其他独立 Required Task 已形成有效结果，则按第三份文档推导为 `partial`。

## 4. Scope Resolution 矩阵

### 4.1 Requested → Resolved

| Requested Scope | Resolution | 裁决 | Strategy | Task | Health | Reason Code |
|---|---|---|---|---|---|---|
| 指定 Document IDs | 全部存在且授权 | 固定版本与 Hash | 只使用 Resolved IDs | 继续 | healthy | 无 |
| “刚上传的文档” | 唯一解析 | 固定唯一资源 | 只使用该资源 | 继续 | healthy | 无 |
| “刚上传的文档” | 多个候选且无法确定 | 不猜测 | 不执行依赖 Scope 的 Step | blocked（若 required） | healthy | `SCOPE_RESOLUTION_FAILED` |
| 指定文档 | 不存在 | 不替换 | 不执行 | blocked/partial | healthy | `SCOPE_RESOLUTION_FAILED` |
| 指定文档 | 未授权 | 不泄露资源是否存在 | 不执行 | blocked | healthy | `SCOPE_RESOLUTION_FAILED` |
| uploaded only | 仅全 KB 有结果 | 禁止扩大到全 KB | 不执行非 Scope 结果 | partial/blocked | healthy | `SCOPE_RESOLUTION_FAILED` |
| Web forbidden | 本地证据不足 | 禁止 Web fallback | 保留 unmet | partial | healthy | `RETRIEVAL_NO_EVIDENCE` |

### 4.2 Executor Preflight

| Plan 时 Scope | Execute 前状态 | 裁决 | Retry | Reason Code |
|---|---|---|---|---|
| 版本/Hash 一致 | 授权有效 | 执行 | 不适用 | 无 |
| 文档版本变化 | 新版本存在 | 不自动切换 | New Run/Contract 才可使用 | `SCOPE_EXECUTION_PRECONDITION_FAILED` |
| 文档被删除 | 不存在 | 不执行 | 仅重新请求/新 Run | `SCOPE_EXECUTION_PRECONDITION_FAILED` |
| 授权失效 | 不可访问 | 不执行 | false | `SCOPE_EXECUTION_PRECONDITION_FAILED` |
| Tool Target 不属于 Resolved Scope | Planner 错误 | 拒绝调用 | 可在当前 Round 修 Strategy，Contract 不变 | `SCOPE_EXECUTION_PRECONDITION_FAILED` |

## 5. Availability 与 Degradation 矩阵

Availability 不修改 Contract。`degraded` 是否满足任务由 `CapabilitySatisfactionPolicy` 和实际产物决定。

| Capability | Runtime 状态 | 实际产物 | Requirement | Strategy/Outcome |
|---|---|---|---|---|
| RAG | available | Direct Evidence + verified citations | required | 正常执行，候选满足 |
| RAG | degraded（reranker failed） | Hybrid Retrieval + Evidence Assessor + citations 成功 | required | 允许降级，System Health degraded，Task 可 completed |
| RAG | degraded（assessor protocol failed） | 只有 provisional evidence | required | 不满足 verified citation；Task partial |
| RAG | available | 正常检索但无证据 | required | 系统 healthy；Capability unsatisfied；Task partial |
| Citation Validation | unavailable | 有 Evidence 但未验证引用 | required | Capability unsatisfied；不得标 validated |
| Calculator | available | 确定性结果完整 | required | satisfied |
| Calculator | degraded | 只返回部分 Scenario | required 全 Scenario | partially_satisfied |
| Router | degraded | Contract 与 Strategy 经 Floor/Extractor/Reconciler 保住 | 不作为业务 Capability | Task 可 completed；Health degraded |
| Guard | protocol_failed | 已验证 Tool/RAG 内容存在 | 按风险策略裁决 | Delivery guard_degraded/rejected；Task 独立判断 |

## 6. Tool Effect 裁决矩阵

### 6.1 Permission 与 Effects

| Permission/Policy | Resolved Effect | 裁决 | Reason Code |
|---|---|---|---|
| Web forbidden | network=external | 拒绝 | `CONTRACT_PERMISSION_CONFLICT` |
| Web forbidden | network=internal | 允许（若 Scope/Policy 允许） | 无 |
| External write forbidden | data=external_write | 拒绝 | `CONTRACT_BLOCKED_BY_POLICY` |
| Side effects forbidden | mutation=reversible/irreversible | 拒绝 | `CONTRACT_PERMISSION_CONFLICT` |
| PII egress forbidden | sensitive_data=pii_egress | 拒绝 | `CONTRACT_BLOCKED_BY_POLICY` |
| Metered calls prohibited/budget zero | cost=metered | 拒绝 | `BUDGET_EXHAUSTED` |
| 任意限制 | resolved effect=unknown | 使用 declared_max_effects 比较 | 视最大风险结果 |

### 6.2 传递闭包

若 Strategy Step A 调用 Tool B，Tool B 内部依赖 Provider C，则最终 Effects 是 A/B/C Effects 的最大闭包。任一传递 Effect 违反 Permission，整个 Step MUST 被拒绝。

| 直接工具 | 间接 Provider | 用户约束 | 裁决 |
|---|---|---|---|
| market_data_tool | external API | 禁止联网 | 拒绝，即使工具名不是 web_search |
| local_rag | local Qdrant | 禁止 Web | 允许 |
| document_parser | local filesystem | 只允许指定文档 | 只允许 Resolved Scope 内文件 |
| calculator | none | 禁止联网 | 允许 |

## 7. Budget 与 Deadline 裁决矩阵

| 状态 | Required 保持 | 新工作 | 已有结果 | Task | Health | Reason Code |
|---|---:|---|---|---|---|---|
| Soft Target 达到 | 是 | 停止低价值额外检索/Repair | 保留 | 按满足度 | healthy | `BUDGET_EXHAUSTED`（warning） |
| Hard Tool Calls 达到 | 是 | 禁止新 Tool Call | 保留 | partial/failed | healthy | `BUDGET_EXHAUSTED` |
| Token Hard Limit 达到 | 是 | 禁止新 LLM Call | 保留 | partial/failed | healthy | `BUDGET_EXHAUSTED` |
| Remaining Time <= Reserved Delivery | 是 | 停止 Replan/Retrieval | 进入 Synthesis/Guard | 按满足度 | healthy | `DEADLINE_EXCEEDED` 或预算原因 |
| Deadline Exceeded | 是 | 禁止新 Planner/RAG/Tool/Round | 保留并封存 | partial/failed | healthy（若正常策略） | `DEADLINE_EXCEEDED` |
| Guard 无剩余预算 | 是 | 不得默认 pass | 可返回与否按风险策略 | 独立 | healthy/degraded | `GUARD_PROTOCOL_DEGRADED` 或 `BUDGET_EXHAUSTED` |

Budget 来自 User Explicit 时属于 `RuntimePolicyEnvelope` 的显式来源，但仍不进入 Business Requirement Contract。

## 8. Strategy Reconciliation 矩阵

| Sealed Contract | Preliminary Strategy | Runtime | Reconciler 动作 | Strategy Status | Reason Code |
|---|---|---|---|---|---|
| RAG required/allowed | 缺少 RAG Step | available+scope valid | 补齐 RAG Step | ready | `STRATEGY_RECONCILED` |
| Calculation required | 只有模型推理 Step | deterministic tool available | 替换/增加 Tool Step | ready | `STRATEGY_RECONCILED` |
| Citation required | 没有 Citation Validation | validator available | 增加 Validation Step | ready | `STRATEGY_RECONCILED` |
| Tool forbidden | 包含 Tool Step | 任意 | 删除 Step；若 required 冲突则 blocked | blocked/ready | `CONTRACT_PERMISSION_CONFLICT` |
| Scope=selected docs | Strategy=all KB | scope valid | 收紧到 Resolved Scope | ready | `STRATEGY_RECONCILED` |
| Scope invalid | 任意依赖 Scope 的 Step | 任意 | 不创建替代 Scope | blocked/unfulfillable | `SCOPE_RESOLUTION_FAILED` |
| Capability required | Strategy 正确 | unavailable | 不伪造 Step 成功；标 unmet | unfulfillable | `CAPABILITY_UNAVAILABLE` |
| Capability optional | Strategy 提议 | Budget 不足 | 删除 optional Step | ready_degraded | `BUDGET_EXHAUSTED` |
| Router failed | Contract 完整 | capability 可用 | Conservative Strategy Builder + Reconcile | ready_degraded | `ROUTER_PROTOCOL_DEGRADED` |
| Extractor failed | Floor 有 explicit required | Router 可用 | Contract 只保 Floor，Strategy 必须满足 Floor | ready_degraded | `EXTRACTOR_PROTOCOL_DEGRADED` |
| Router+Extractor failed | Floor 有 required | 可用 | Safe deterministic fallback，仅满足 Floor | ready_degraded | 两个 degraded codes |
| Router+Extractor failed | Floor 空 | 无可靠意图 | 仅低风险直接回答或澄清，不启用高风险 Capability | blocked/ready_degraded | `STRATEGY_UNCLASSIFIED_CONTROL_STATE` |

## 9. DAG 与 Execution Round 规则

Strategy Step 的 `depends_on` MUST 形成无环图。互不依赖的 Step MAY 并行；依赖 Step 必须等待前置成功产物。

| 情况 | 裁决 |
|---|---|
| Plan Review 要求修订 | 当前 Execution Round 内 Repair；不增加 Round |
| Execute→Observe→Validate 完成且仍有可处理未完成项 | MAY 进入下一 Execution Round |
| Assessor/Guard 协议修复 | 同节点 Protocol Repair；不增加 Execution Round |
| Required Capability unavailable/forbidden/scope invalid | 不用空 Round 重试；按 Retryability 决定等待或结束 |
| Deadline/Budget 不允许下一轮 | Assemble 已验证结果，记录 unmet |
| DAG cycle | Strategy invalid，禁止执行 | 

## 10. Idempotency / Replay 裁决矩阵

| Existing Step Record | Freshness | Side Effect 能力 | 裁决 | Reason Code |
|---|---|---|---|---|
| succeeded，身份字段全部一致 | valid | 任意 | reuse；不重新执行 | 无 |
| succeeded | expired | 只读确定性 | 按 Freshness Policy 重算 | 无 |
| succeeded | expired | 外部副作用 | 不按 TTL 重放；查 Provider 状态 | `TOOL_RESULT_UNKNOWN`（若无法确认） |
| running | 任意 | 任意 | 等待/接管 Lease；不得并行重复执行 | 无 |
| failed_retryable | 任意 | 无副作用或 Provider 幂等 | 预算内重试 | `TOOL_EXECUTION_FAILED` |
| failed_terminal | 任意 | 任意 | 不重试 | `TOOL_EXECUTION_FAILED` |
| unknown | 任意 | 副作用 | 只允许 Status Query/Reconciliation | `TOOL_RESULT_UNKNOWN` |
| cancelled before dispatch | 任意 | 任意 | 可在 New Run 重试 | `RUN_CANCELLED` |
| cancelled after dispatch | 任意 | 副作用 | 查询状态；不得假定未发生 | `TOOL_RESULT_UNKNOWN` |
| Contract/Scope/Tool Version 任一不一致 | 任意 | 任意 | 禁止 reuse | 无 |

Request Idempotency、Step Idempotency 和 Provider Idempotency MUST 分层处理。无 Provider Idempotency 时不得宣称 exactly-once。

## 11. Cancellation 与 Late Result 裁决

| 事件 | 新调用 | 在途调用 | 成功结果 | 最终状态 |
|---|---|---|---|---|
| Client Cancel | 禁止 | 协作式取消 | 已完成结果保留 | 按完成事实 + `RUN_CANCELLED` |
| Deadline Exceeded | 禁止 | 尽力取消 | 已完成结果保留 | partial/failed |
| Tool 在 Cutoff 后成功返回 | 不启动后续依赖 | 记录 Late Result | 不修改已封存状态 | `LATE_RESULT_RECORDED` |
| 副作用 Tool 在 Cancel 后未知 | 禁止重放 | Status Query | UNKNOWN 审计 | `TOOL_RESULT_UNKNOWN` |
| 可补偿操作已成功 | 不自动假定撤销 | 仅显式 Compensation Policy 可执行 | 原结果保留 | 按补偿结果另记事件 |

## 12. Clarification、Blocked 与 Safe Fallback

| 原因 | 是否澄清 | 是否可执行其他独立任务 | Task 预期 |
|---|---:|---:|---|
| 同一用户 Authority 的 required/forbidden 真冲突 | 是 | 否，若冲突决定执行路径 | blocked |
| 系统 Hard Policy 禁止用户 Required | 否 | 可执行不受影响的独立任务，否则 blocked | blocked/partial |
| Scope 多候选且 required | 是 | 可执行不依赖 Scope 的 Tool | blocked/partial |
| Capability unavailable 且不可重试 | 否 | 是 | partial/failed |
| 非关键参数缺失，可用 Scenario 表达 | 否 | 是 | 继续 |
| Router 协议失败但 Contract 完整 | 否 | 是 | 继续，Health degraded |

## 13. 矩阵完备性要求

实现前 MUST 将本矩阵转换为表驱动测试，至少证明：

- 每个 `Requirement × Permission × Scope × Availability` 组合均有结果。
- 每个结果都有稳定 Reason Code 或明确“无需 Reason Code”。
- Forbidden Violation 与 Silent Scope Expansion 不存在任何允许路径。
- Router/Extractor fallback 不得降低 Floor。
- `unknown` 副作用不存在自动重放路径。
- 未分类组合统一 fail closed，而不是由 LLM 自由决定。
