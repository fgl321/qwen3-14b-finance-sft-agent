# 03 Capability Satisfaction 与状态推导矩阵

> 状态：冻结候选设计的规范性状态表  
> 文档版本：`satisfaction-status-matrix-v1`  
> 依赖：`01_control_plane_schema.md`、`02_constraint_strategy_matrix.md`  
> 目标：从事实机械推导 `task_status`、`system_health`、`delivery_status`

## 1. 三层状态是独立 Source of Truth

```text
Task Status
→ 用户 Required Deliverables 实际完成了多少

System Health
→ 运行组件是否按设计正常工作

Delivery Status
→ 最终内容是否通过所需交付验证、是否允许返回
```

三者禁止互相替代。典型合法组合：

```text
completed + degraded + validated
partial   + healthy  + validated_with_limitations
completed + degraded + guard_degraded
blocked   + healthy  + not_generated
```

## 2. Status Assembler 输入事实

业务节点只能写事实，不能写最终状态。Status Assembler MUST 只消费：

- `SealedEffectiveContract.task_requirements`
- `CapabilityOutcome[]`
- `StepExecutionRecord[]`
- `RagStageStatus` 与 Citation Validation Outcome
- `RuntimeCapabilitySnapshot` 与实际运行错误
- `CancellationState`、Deadline、Budget 使用事实
- `DeliveryGuardOutcome`
- `RequestRiskClass`
- Reason Code Registry

节点禁止直接写：

```text
task_status
system_health
delivery_status
overall_status
```

## 3. Capability Satisfaction Policy

每项 Capability MUST 有版本化 Policy：

```text
acceptable_runtime_statuses
required_outputs
minimum_quality
allowed_degradations
```

Satisfaction 先检查产物，再参考组件状态：

```text
Required Outputs 完整且通过验证 → satisfied
部分有效，但没有满足全部 Required Deliverable → partially_satisfied
没有形成有效必需产物 → unsatisfied
Contract 未要求 → not_required
```

组件 `degraded` 不自动等于 Capability unsatisfied；组件 `healthy` 也不保证业务有证据。

## 4. 核心 Capability Satisfaction 矩阵

### 4.1 Knowledge Retrieval

| Retrieval | Rerank | Assessor | Evidence | Required Citation | Outcome | Health | Reason Code |
|---|---|---|---|---:|---|---|---|
| completed | completed | completed | Direct + verified citations | 是 | satisfied | healthy | 无 |
| completed | failed/degraded | completed | Hybrid 证据充分 + verified citations | 是 | satisfied | degraded | `CAPABILITY_DEGRADED` |
| completed | completed | protocol_failed | provisional only | 是 | unsatisfied | degraded | `CAPABILITY_DEGRADED` |
| completed | completed | completed | Partial evidence，用户允许部分支持 | 否/Preferred | satisfied 或 partial，按 Task Policy | healthy | `RETRIEVAL_NO_EVIDENCE`（若不足） |
| completed | completed | completed | 正常检索但无相关证据 | 是 | unsatisfied | healthy | `RETRIEVAL_NO_EVIDENCE` |
| service_failed | not_run | not_run | 无 | 是 | unsatisfied | degraded/failed | `CAPABILITY_UNAVAILABLE` |
| not_attempted | not_run | not_run | 无 | 是 | unsatisfied | healthy 或 degraded（看 Strategy 原因） | `CONTRACT_REQUIRED_CAPABILITY_MISSING` |

关键语义：正常检索但无证据可使 Task partial，但 System 仍 healthy；检索到候选而 Assessor 协议失败时，provisional 不满足 verified citations，System degraded。

### 4.2 Financial Calculation

| Tool Execution | Required Outputs | Validation | Outcome | Health | Reason Code |
|---|---|---|---|---|---|
| succeeded | 全部场景/字段完整 | passed | satisfied | healthy | 无 |
| succeeded | 部分 Scenario | passed | partially_satisfied | healthy | `CONTRACT_REQUIRED_CAPABILITY_MISSING` |
| succeeded | 字段完整 | semantic validation failed | unsatisfied/partial | healthy | `TOOL_EXECUTION_FAILED` |
| failed_terminal | 无 | 不适用 | unsatisfied | healthy/degraded（按失败类型） | `TOOL_EXECUTION_FAILED` |
| unavailable | 无 | 不适用 | unsatisfied | degraded | `CAPABILITY_UNAVAILABLE` |
| reused success | 全部、Freshness valid | passed | satisfied | healthy | 无 |
| unknown side effect | 无法确认 | 不允许重复 | unsatisfied | degraded | `TOOL_RESULT_UNKNOWN` |

LLM 心算、通用知识或文档经验规则不得替代 Required Deterministic Tool Output。

### 4.3 Citation Validation

| Citations | IDs/Scope | Claim Mapping | Validator | Outcome | Delivery 约束 |
|---|---|---|---|---|---|
| 存在 verified citations | 全部合法 | 语义一致 | passed | satisfied | 可 validated |
| 存在 citations | 有非法 ID/越 Scope | 任意 | failed | unsatisfied | rewrite/rejected |
| 存在 citations | 合法 | 概念映射错误 | failed | unsatisfied | rewrite/rejected |
| 无 verified citations | 不适用 | 文档 Required | 正常确认缺失 | unsatisfied | 只允许明确未完成，不得通用知识替代 |
| provisional citations only | Scope 合法 | 未 Assessed | protocol degraded | unsatisfied | 不得把文档事实标 validated |
| Validator unavailable | Evidence 存在 | 未验证 | unavailable | unsatisfied | guard_degraded/rejected，按风险 |

### 4.4 Complex Reasoning / Synthesis

| 输入事实 | Required Capabilities | Synthesis | Outcome |
|---|---|---|---|
| Required Inputs 全部 satisfied | 全部完成 | 成功且来源区分正确 | satisfied |
| Required Inputs 部分完成 | 部分 unmet | 明确 Partial/失败项 | partially_satisfied |
| Required Inputs 缺失 | 仍输出确定性完整结论 | Provenance 违规 | unsatisfied |
| Tool/RAG 成功但 Synthesis 服务失败 | 事实存在，无法形成答案 | failed | unsatisfied |

## 5. Task Requirement 聚合

| Task 所需 Capability | 聚合结果 | Task Requirement Outcome |
|---|---|---|
| 全部 satisfied | 全满足 | satisfied |
| 至少一个 partially_satisfied，其他 satisfied | 部分 | partially_satisfied |
| 任一 required Capability unsatisfied | 未满足 | unsatisfied |
| Task.required=false | 不参与 Task Status 必需集合 | optional outcome |

Task 依赖 DAG 规则：下游 Derived Task 只有在其 `depends_on` 产物满足 Policy 时才可 satisfied。下游自然语言声称完成不能反向弥补上游缺失。

## 6. Task Status 推导

严格定义：

- `blocked`：在任何业务执行合法开始前，被 Policy、Permission、Scope 或不可解决 Contract Conflict 阻止。
- `failed`：合法开始执行，但没有任何 Required Task 形成可交付有效结果。
- `partial`：至少一个 Required Task 已形成有效结果，同时至少一个 Required Task 未满足。
- `completed`：所有 Required Task 均 satisfied。

### 6.1 推导表

| 执行是否合法开始 | Required Task satisfied 数 | Required Task unmet 数 | 裁决 |
|---:|---:|---:|---|
| 否 | 0 | >=1 | blocked |
| 是 | 0 | >=1 | failed |
| 是 | >=1 | >=1 | partial |
| 是/无需执行 | 全部 | 0 | completed |

若先完成不受影响的独立 Required Task，随后另一个 Required Task 因 Scope/Policy 不能执行，最终是 `partial`，不再是 `blocked`。

### 6.2 常见案例

| 事实 | Task Status | Primary Reason |
|---|---|---|
| Tool ✅、RAG ✅、Citation ✅ | completed | 无 |
| Tool ✅、Required RAG 正常但无证据 | partial | `RETRIEVAL_NO_EVIDENCE` |
| Tool ✅、RAG 技术故障 | partial | `CAPABILITY_UNAVAILABLE` |
| 所有 Required Tool 均失败，无有效结果 | failed | `TOOL_EXECUTION_FAILED` |
| Required Scope 在执行前无法解析，没有其他 Required Task | blocked | `SCOPE_RESOLUTION_FAILED` |
| 系统 Policy 禁止唯一 Required Capability | blocked | `CONTRACT_BLOCKED_BY_POLICY` |
| Budget 耗尽前完成部分 Required Tasks | partial | `BUDGET_EXHAUSTED` |
| Deadline 到达且没有任何 Required Task 完成 | failed | `DEADLINE_EXCEEDED` |
| 用户取消前完成部分 Required Tasks | partial | `RUN_CANCELLED` |

## 7. System Health 推导

System Health 只描述组件运行，不描述是否找到业务证据。

| 事件 | Health 影响 |
|---|---|
| 正常检索但无证据 | healthy |
| Policy 正常阻止操作 | healthy |
| 用户 Constraint Conflict | healthy |
| Budget 正常达到 Hard Limit | 通常 healthy |
| Deadline 按配置到达 | 通常 healthy |
| Router Protocol Repair 成功 | 可保持 healthy，记录 repaired metric |
| Router Protocol 最终失败但 fallback 生效 | degraded |
| Extractor Protocol 最终失败 | degraded |
| Reranker 失败但 RAG 降级可用 | degraded |
| Assessor/Guard Protocol 最终失败 | degraded |
| 单个非关键 Provider 不可用 | degraded |
| Required 核心服务全部不可用 | failed |
| Audit/Integrity Gate 无法运行 | failed；高风险执行不得继续 |

| Critical Components | Non-critical Components | System Health |
|---|---|---|
| 全正常 | 全正常 | healthy |
| 全正常 | 任一 degraded/failed，存在安全降级 | degraded |
| 任一 Critical failed 且无法安全继续 | 任意 | failed |

Critical 的定义属于 Runtime Policy，至少包含 Policy/Permission Enforcement、Contract Integrity、Scope Authorization 和高风险 Delivery Gate。

## 8. Delivery Status 推导

Delivery 由 `RequestRiskClass × Guard State × Verified Content Available` 决定。

```text
low：普通知识解释、非个性化低风险问答
medium：个性化金融分析、资产/保险/贷款规划，但不执行资金动作
high：交易、转账、外部写操作、不可逆动作或强监管高风险输出
```

| Risk | Guard | Verified Content | Delivery Status | 可返回内容 |
|---|---|---|---|---|
| low | passed | 任意合法 | validated | 完整合法答案 |
| medium | passed | Required 内容满足 | validated | 完整答案 |
| 任意 | passed_with_limitations | 有部分已验证内容 | validated_with_limitations | 已验证部分 + 明确未完成项 |
| low | protocol_degraded | 有确定性/已验证内容 | guard_degraded | 可有限返回，明确 Guard 降级 |
| medium | protocol_degraded | 有确定性 Tool/RAG subset | guard_degraded | 只返回 verified subset + limitation |
| medium | protocol_degraded | 只有未验证模型结论 | rejected/not_generated | 不返回确定性建议 |
| high | protocol_degraded/not_run | 任意 | rejected/not_generated | 不执行/不交付高风险内容 |
| 任意 | rejected | 任意 | rejected | 安全拒绝或无答案 |
| 任意 | passed | 无法生成答案 | not_generated | 错误说明 |

`guard_degraded` 永远不得显示为 `validated`。时间或预算不足不能默认 Guard Pass。

## 9. 三层组合矩阵

| Task | Health | Delivery | 语义 |
|---|---|---|---|
| completed | healthy | validated | 正常完整完成 |
| completed | degraded | validated | 任务完成，内部组件降级但最终已验证 |
| completed | degraded | guard_degraded | 任务材料完成，交付验证降级 |
| partial | healthy | validated_with_limitations | 系统正常，但业务证据/信息不足 |
| partial | degraded | validated_with_limitations | 部分任务完成且组件降级 |
| partial | degraded | guard_degraded | 仅允许返回已验证子集 |
| failed | healthy | validated_with_limitations/not_generated | 系统正常执行但无有效业务结果 |
| failed | failed | not_generated | 系统故障导致失败 |
| blocked | healthy | not_generated | Policy/Permission/Scope/Conflict 阻止执行 |
| blocked | degraded | not_generated | 控制面降级且无法安全解析执行条件 |

## 10. Required Citation 与通用知识替代

当 Task `requires_citations=true`：

- Verified Citation 缺失时，Document-required Claim MUST NOT 由模型参数记忆替代。
- Tool-supported 独立部分 MAY 正常返回。
- 最终答案 MUST 区分“正常检索无证据”和“检索/审核服务异常”。
- Provisional Evidence 只用于审计和候选展示，不满足 Citation Validation。

| RAG 事实 | 正确表述 | 错误表述 |
|---|---|---|
| 正常检索无证据 | 已完成检索，但当前文档没有足够直接证据 | 检索服务异常 |
| Retriever 成功，Assessor 协议失败 | 检索成功，但证据审核协议异常，候选尚未验证 | 没有搜到任何资料 |
| Retrieval 服务失败 | 检索服务异常 | 文档中没有相关内容 |

## 11. Cancellation、Late Result 与状态封存

| 状态封存前事实 | 推导 |
|---|---|
| Cancel 前全部 Required satisfied | completed + `RUN_CANCELLED` 审计 |
| Cancel 前部分 satisfied | partial |
| Cancel 前无 satisfied、已合法开始 | failed |
| Cancel 在业务执行前生效 | blocked |

Run Status 一旦封存，Late Result 只能增加 Audit Event 与复用候选，禁止修改 Task/Health/Delivery。New Run MAY 在验证身份与 Freshness 后复用 Late Result。

## 12. Budget 与状态

| Budget 事实 | Task | Health | Delivery |
|---|---|---|---|
| Hard Limit 到达但全部 required 已完成 | completed | healthy | 正常按 Guard |
| Hard Limit 到达且部分 required 完成 | partial | healthy | validated_with_limitations/按 Guard |
| Hard Limit 到达且无 required 完成 | failed | healthy | not_generated/限制说明 |
| Budget 子系统故障错误拒绝合法预算 | 按产物 | degraded | 按 Guard |

预算不足是 Runtime Policy 结果，不应默认标记 System degraded。

## 13. Reason Code 选择规则

- `reason_codes` 包含全部稳定原因。
- `primary_reason_code` 表示最直接决定 Task/Delivery 最终状态的原因。
- Health-only 降级不应覆盖更重要的 Task Primary Reason。

优先顺序：

```text
Policy/Permission Block
> Scope/Contract Conflict
> Required Capability Unavailable
> Execution Terminal Failure
> No Evidence / Unmet Output
> Deadline/Budget/Cancellation
> Non-critical Component Degradation
```

示例：Tool 成功、Required RAG 无证据、Router degraded：

```text
task_status = partial
system_health = degraded
primary_reason_code = RETRIEVAL_NO_EVIDENCE
reason_codes = [RETRIEVAL_NO_EVIDENCE, ROUTER_PROTOCOL_DEGRADED]
```

## 14. Legacy Overall Status 映射

| Task Status | Delivery Status | legacy_overall_status |
|---|---|---|
| completed | validated | completed |
| completed | validated_with_limitations/guard_degraded | completed_with_limitations |
| partial | 非 rejected | partial |
| failed | 任意 | failed |
| blocked | 任意 | blocked |
| 任意 | rejected/not_generated | Task completed 时为 delivery_rejected，否则沿用 Task |

`system_health=degraded` 单独展示，不应把业务 `completed` 改成 `partial`。

## 15. Status Assembler 不变量

1. Task Status 只看 Required Task 实际完成事实。
2. System Health 只看组件运行事实。
3. Delivery Status 只看交付验证和风险策略。
4. 正常无证据不是系统故障。
5. Router degraded 但任务全部完成，不得产生假 partial。
6. Guard degraded 不得显示 validated。
7. Budget/Deadline 不得降低 Required。
8. Late Result 不得修改封存状态。
9. Provisional Evidence 不得满足 Required Citation。
10. 未分类组合 MUST fail closed，并产生稳定 Reason Code。
