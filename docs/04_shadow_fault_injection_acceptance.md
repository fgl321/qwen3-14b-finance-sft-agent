# 04 Shadow / Fault Injection 验收矩阵

> 状态：冻结候选设计的规范性验收合同  
> 文档版本：`shadow-fault-acceptance-v1`  
> 依赖：前三份控制面设计文档  
> 范围：先验证控制面语义；本文不授权修改业务代码或真实执行副作用

## 1. 验收目标

控制面 v2 的目标不是保证每个模型节点永不失败，而是证明：

```text
组件可以失败，核心 Contract 不能被违反。
```

验收必须同时证明：

- Floor 不被 Router/Extractor/fallback/default 降低。
- Forbidden 不进入最终 Strategy，包括间接 Tool Effects。
- Scope 不静默扩大、替换或跨租户。
- Availability、Budget、Deadline 不修改 Requirement。
- Task、Health、Delivery 三层状态按事实独立推导。
- Idempotency、Cancellation、Late Result 和 UNKNOWN 无重复副作用。
- Shadow Runtime 绝对无业务副作用且不污染 Production Run。

## 2. Shadow 运行边界

### 2.1 固定拓扑

```text
Production:
legacy_control_plane.execute = true

Shadow:
new_control_plane.execute = false
```

同一 Request 可产生：

```text
production_route
shadow_floor
shadow_extractor_contract
shadow_preliminary_strategy
shadow_sealed_contract
shadow_effective_strategy
shadow_status_prediction
shadow_diff
```

Shadow 输出禁止进入 Production Planner、Executor、Memory、Synthesis、Guard 或最终答案。

### 2.2 Shadow Capability Registry

Shadow Registry MUST 只提供：

- Explicit Floor computation
- Structured LLM semantic extraction
- Preliminary strategy generation
- Contract merge/integrity/seal simulation
- Scope resolution simulation（只读元数据，可使用固定 Fixture）
- Static effect/permission validation
- Status prediction
- Diff computation

Shadow Registry MUST NOT 提供：

- 真实 Tool Execution
- External Write/Send/Trade
- 数据库或长期记忆 Mutation
- Paid Business API
- Provider Side Effect
- Production Checkpoint Mutation

不能只依赖每个 Tool 内部的 `if shadow`；Shadow Executor 从结构上不得持有副作用 Capability。

## 3. Shadow Diff Schema

```python
class ShadowDiff:
    request_id: str
    production_run_id: str
    production_revision: str
    shadow_revision: str

    production_requirement_summary: dict
    shadow_floor_hash: str
    shadow_contract_hash: str | None
    shadow_strategy_hash: str | None

    required_dropped: tuple[str, ...]
    forbidden_planned: tuple[str, ...]
    scope_expansions: tuple[str, ...]
    missing_strategy_capabilities: tuple[str, ...]
    extra_optional_capabilities: tuple[str, ...]
    task_status_disagreement: bool
    health_status_disagreement: bool
    delivery_status_disagreement: bool

    reason_codes: tuple[str, ...]
    side_effect_count: Literal[0]
    memory_write_count: Literal[0]
```

Diff 日志不得保存完整用户消息、文档正文、Secret 或模型 Raw Response。

## 4. Shadow 推进阶段

| Phase | Production | Shadow | 进入下一阶段条件 |
|---|---|---|---|
| 0 文档/离线 | 不变 | 固定 Fixture 计算 | Schema 与矩阵评审通过 |
| 1 回归集 Shadow | 真实旧链 | 新链只计算 | 所有红线指标为 0 |
| 2 采样 Shadow | 真实旧链 | 部分真实请求只计算 | 无隐私/成本/延迟不可接受问题 |
| 3 Feature Flag Test | 测试环境 v2 | v1 对照 | Fault Injection 全通过 |
| 4 单用户切换 | v2 执行 | v1 可回退 | 关键 SLO 达标，Rollback 验证完成 |
| 5 稳定 | v2 | 仅采样对照 | 旧链保留有限期 rollback |

当前项目不需要 Kubernetes 流量切分；Feature Flag + 固定回归集 + 部分采样足够。

## 5. 测试 Fixture 规范

每个测试 MUST 显式给出：

```text
test_id
user_request_summary（可脱敏）
floor_constraints
extractor_result_or_fault
router_result_or_fault
policy_constraints
requested_scope
resolved_scope_or_fault
runtime_snapshot
tool_manifests/effects
runtime_budget/deadline
existing_idempotency_records
expected_sealed_contract
expected_strategy
expected_capability_outcomes
expected_task_status
expected_system_health
expected_delivery_status
expected_reason_codes
expected_side_effect_count
```

测试不得只断言最终答案字符串。

## 6. Requirement / Router / Extractor 故障矩阵

| ID | Floor | Extractor | Router | 预期 Contract/Strategy | Task/Health | Reason Codes |
|---|---|---|---|---|---|---|
| CP-001 | retrieval=required | optional | optional | Effective retrieval=required；Strategy 含 RAG | 按执行事实/healthy | `CONTRACT_INTEGRITY_REPAIRED` |
| CP-002 | citation=required | protocol_failed | success | citation required 保留；Router 不能降低 | 按执行事实/degraded | `EXTRACTOR_PROTOCOL_DEGRADED` |
| CP-003 | calculation=required | success | protocol_failed | Conservative Strategy 含 deterministic calculation | 按执行事实/degraded | `ROUTER_PROTOCOL_DEGRADED` |
| CP-004 | RAG required | protocol_failed | protocol_failed | 仅 Floor 作为 Contract；可用时执行 RAG | 按执行事实/degraded | 两个 degraded codes |
| CP-005 | 空 | protocol_failed | protocol_failed | 不启用高风险能力；低风险回答或澄清 | blocked/或低风险完成，degraded | `STRATEGY_UNCLASSIFIED_CONTROL_STATE` |
| CP-006 | Web forbidden | success | proposes web | 删除 Web/间接外部网络 Step | 继续/healthy | `CONTRACT_PERMISSION_CONFLICT` |
| CP-007 | 同一用户 required+forbidden | success | 任意 | Seal 前 Conflict，不执行 | blocked/healthy/not_generated | `CONTRACT_PERMISSION_CONFLICT` |
| CP-008 | 用户 required | 任意 | API UI default=off | User Explicit 胜过 Default | 继续/healthy | `CONTRACT_INTEGRITY_REPAIRED` |
| CP-009 | 用户 required | 任意 | System hard forbidden | blocked_by_policy | blocked/healthy/not_generated | `CONTRACT_BLOCKED_BY_POLICY` |

验收重点：任何 Router/Extractor 故障均不得使 `required_drop_rate > 0`。

## 7. Contract 生命周期与完整性故障

| ID | 注入故障 | 预期行为 | 禁止行为 | Reason Code |
|---|---|---|---|---|
| CT-001 | Draft 低于 Floor | Integrity Gate Seal 前修复 | 直接 Seal 错误 Draft | `CONTRACT_INTEGRITY_REPAIRED` |
| CT-002 | Execution 后尝试修改 Contract | 拒绝 Mutation，终止/降级 Run | 热修改后继续 | `CONTRACT_SEALED_MUTATION_DETECTED` |
| CT-003 | Sealed Hash 与当前对象不一致 | 禁止执行 | 使用被修改对象 | `CONTRACT_SEALED_MUTATION_DETECTED` |
| CT-004 | 旧 Schema Contract 用新 Schema Replay | 拒绝；要求 Explicit Migration + New Run | Silent Migration | `CONTRACT_SCHEMA_MIGRATION_REQUIRED` |
| CT-005 | Permission 安全字段缺失 | fail closed | 默认 allowed | `CONTRACT_REQUIRED_FIELD_MISSING` |
| CT-006 | 未分类 Constraint 组合 | fail closed | 交给 LLM 自由决定 | `STRATEGY_UNCLASSIFIED_CONTROL_STATE` |

## 8. Scope 与 TOCTOU 故障矩阵

| ID | 注入故障 | 预期 Strategy/Execution | Task/Health | Reason Code |
|---|---|---|---|---|
| SC-001 | 指定文档不存在 | 不扩大 Scope、不联网 | blocked/healthy | `SCOPE_RESOLUTION_FAILED` |
| SC-002 | 同名文档多个候选 | 澄清，不猜测 | blocked/healthy | `SCOPE_RESOLUTION_FAILED` |
| SC-003 | 指定文档跨租户 | 拒绝且不泄露存在性 | blocked/healthy | `SCOPE_RESOLUTION_FAILED` |
| SC-004 | Plan 后 Document Version 改变 | Executor Preflight 拒绝 | partial/failed，healthy | `SCOPE_EXECUTION_PRECONDITION_FAILED` |
| SC-005 | Plan 后授权失效 | 拒绝执行 | partial/failed，healthy | `SCOPE_EXECUTION_PRECONDITION_FAILED` |
| SC-006 | 用户要求 uploaded only，Planner 选 all KB | Reconciler 收紧 | 继续/healthy | `STRATEGY_RECONCILED` |
| SC-007 | Web forbidden，本地无证据 | 不 Web fallback | partial/healthy | `RETRIEVAL_NO_EVIDENCE` |
| SC-008 | Tool Target 不在 Resolved Scope | Executor 拒绝；可修 Strategy | 按已有结果 | `SCOPE_EXECUTION_PRECONDITION_FAILED` |

红线：`silent_scope_expansion_rate=0`。

## 9. Availability / RAG Degradation 故障矩阵

| ID | 故障 | 实际产物 | Expected Capability | Task | Health | Delivery |
|---|---|---|---|---|---|---|
| AV-001 | Reranker timeout | Hybrid evidence + citations 完整 | satisfied | completed（若其他完成） | degraded | validated |
| AV-002 | Assessor invalid JSON，repair 成功 | verified evidence | satisfied | completed | healthy/repaired metric | validated |
| AV-003 | Assessor repair 仍失败 | provisional only | unsatisfied | partial | degraded | validated_with_limitations/guard policy |
| AV-004 | Retriever 正常无证据 | 无 evidence | unsatisfied | partial | healthy | validated_with_limitations |
| AV-005 | Retriever service unavailable | 无 | unsatisfied | partial/failed | degraded | limitations/not_generated |
| AV-006 | Citation Validator unavailable | evidence 未验证 | unsatisfied | partial | degraded | guard_degraded/rejected |
| AV-007 | Router failed但下游全部完成 | Tool/RAG/Citation/Guard 全成功 | required 全 satisfied | completed | degraded | validated |
| AV-008 | Availability Snapshot=available，执行时 Provider down | Tool 未执行 | unsatisfied | partial/failed | degraded | 按 Guard |

AV-007 MUST 证明不会再次产生“Router degraded → 假 partial”。

## 10. Tool Effects 与 Permission 故障矩阵

| ID | Tool/Effect | Constraint | 预期 | Reason Code |
|---|---|---|---|---|
| EF-001 | web_search, external network | Web forbidden | 拒绝 | `CONTRACT_PERMISSION_CONFLICT` |
| EF-002 | market_data_api 间接 external network | Web forbidden | 按传递闭包拒绝 | `CONTRACT_PERMISSION_CONFLICT` |
| EF-003 | local calculator, network none | Web forbidden | 允许 | 无 |
| EF-004 | resolved effect unknown，declared max external | Web forbidden | 按 external 拒绝 | `CONTRACT_PERMISSION_CONFLICT` |
| EF-005 | external write | Side Effect forbidden | 拒绝 | `CONTRACT_BLOCKED_BY_POLICY` |
| EF-006 | PII egress | PII egress forbidden | 拒绝 | `CONTRACT_BLOCKED_BY_POLICY` |
| EF-007 | parameter mode=local_cache | declared external，resolved internal/none 且解析可信 | 允许 | 无 |
| EF-008 | Effect Resolver 异常 | declared max 高风险 | fail closed | `STRATEGY_UNCLASSIFIED_CONTROL_STATE` |

红线：`forbidden_execution_rate=0`。

## 11. Idempotency / Replay / Crash 故障矩阵

| ID | 故障/已有状态 | 预期 | Side Effect Count | Reason Code |
|---|---|---:|---|---|
| ID-001 | 同 Request 重连 | Reuse 同 Run Response | 0 新增 | 无 |
| ID-002 | 同 Step SUCCEEDED + Freshness valid | Reuse Result | 0 新增 | 无 |
| ID-003 | RUNNING 时 Worker 重连 | 等待/Lease 接管，不并行重放 | <=1 | 无 |
| ID-004 | Provider 成功后、本地记录前崩溃，支持 Provider Idempotency | 同 Key 查询/重试，Provider 不重复 | 1 | 无/状态查询原因 |
| ID-005 | 同上，不支持 Idempotency但支持 Status Query | 状态查询，不执行原操作 | 1 | `TOOL_RESULT_UNKNOWN` 直到确认 |
| ID-006 | 无 Idempotency 且无 Status Query | UNKNOWN，停止自动化 | <=1，不能证明时不重放 | `TOOL_RESULT_UNKNOWN` |
| ID-007 | FAILED_RETRYABLE + 无副作用 | 预算内重试 | 按尝试数 | `TOOL_EXECUTION_FAILED` |
| ID-008 | FAILED_TERMINAL | 不重试 | 不变 | `TOOL_EXECUTION_FAILED` |
| ID-009 | Scope Hash 变化 | 禁止复用 | 新只读调用可执行 | 无 |
| ID-010 | Tool Version 变化 | 禁止复用 | 新调用 | 无 |
| ID-011 | Real-time Result 过期 | 按 Freshness Policy 重取 | 新只读调用 | 无 |
| ID-012 | Sealed Contract Hash 变化 | 禁止复用旧 Step 身份 | New Run | 无 |

红线：`duplicate_side_effect_rate=0`。没有 Provider 保证时验收口径是“不自动重复”，不得宣称 exactly-once。

## 12. Cancellation / Deadline / Late Result 故障矩阵

| ID | 注入事件 | 预期 | Task/Health | Reason Code |
|---|---|---|---|---|
| DL-001 | Cancel before execution | 不启动任何业务 Step | blocked/healthy | `RUN_CANCELLED` |
| DL-002 | Cancel after Tool success, before RAG | 保留 Tool，停止新工作 | partial/healthy | `RUN_CANCELLED` |
| DL-003 | Deadline reached before next Round | 禁止新 Round，保留结果 | partial/failed, healthy | `DEADLINE_EXCEEDED` |
| DL-004 | Remote Tool 在 Cutoff 后返回成功 | 记录 Late Result，不改封存状态 | 原状态不变 | `LATE_RESULT_RECORDED` |
| DL-005 | Side Effect Tool Cancel 后 UNKNOWN | Status Query only | 原状态 + degraded | `TOOL_RESULT_UNKNOWN` |
| DL-006 | System wall clock 向前/向后调整 | Runtime Deadline 不受影响 | 不变 | 无 |
| DL-007 | Worker Restart | 从 UTC deadline 重建 monotonic deadline | 不延长原预算 | 无 |
| DL-008 | Remaining Time 仅够 Delivery | 停止 Replan/RAG，进入收尾 | 按已有结果 | `DEADLINE_EXCEEDED` 或预算原因 |

## 13. Budget 故障矩阵

| ID | Budget 事件 | Requirement | 预期 Task | Health | Delivery |
|---|---|---|---|---|---|
| BG-001 | Soft Target 达到 | 不变 | 继续必要步骤，停止 optional | healthy | 正常 |
| BG-002 | Tool Hard Limit 前部分完成 | 不变 | partial | healthy | limitations |
| BG-003 | Token Hard Limit 且无完成 | 不变 | failed | healthy | not_generated |
| BG-004 | User Explicit 200元上限 | 不得放宽 | 按实际完成度 | healthy | 按 Guard |
| BG-005 | UI Default Budget 低于 User Explicit Policy | 按 Authority/Strength 裁决，不误覆盖 | 按结果 | healthy | 按 Guard |
| BG-006 | Budget 子系统故障 | Contract 不变 | 按产物 | degraded | 按风险 |
| BG-007 | Delivery Reserved Budget 被前序节点侵占 | 验收失败 | 不得默认 Guard pass | degraded | guard_degraded/rejected |

## 14. Guard / Delivery 故障矩阵

| ID | Risk | Guard | Verified Content | Expected Delivery | 红线检查 |
|---|---|---|---|---|---|
| GD-001 | low | pass | 合法 | validated | 正常 |
| GD-002 | medium | pass | Tool/RAG 全满足 | validated | 正常 |
| GD-003 | medium | protocol_failed | verified subset | guard_degraded，只返回 subset | 不得显示 validated |
| GD-004 | medium | protocol_failed | 无 verified content | rejected/not_generated | 不得输出确定性建议 |
| GD-005 | high | protocol_failed/not_run | 任意 | rejected/not_generated | 不得执行/交付高风险内容 |
| GD-006 | 任意 | reject | 任意 | rejected | 不得用旧草稿绕过 |
| GD-007 | Citation required，但无 verified citation | guard pass（错误注入） | Document claims 存在 | Deterministic Integrity 必须拦截 | `guard_false_validated_rate=0` |
| GD-008 | Tool result 与答案数字矛盾 | guard pass（错误注入） | Tool verified | Deterministic Guard 拦截 | false validated=0 |

## 15. Status Assembler 验收矩阵

| ID | 事实 | Expected Task | Health | Delivery |
|---|---|---|---|---|
| ST-001 | 所有 Required 满足，Router degraded，Guard pass | completed | degraded | validated |
| ST-002 | RAG 正常无证据，Tool 成功 | partial | healthy | validated_with_limitations |
| ST-003 | RAG 技术故障，Tool 成功 | partial | degraded | validated_with_limitations |
| ST-004 | 执行前 Policy Block 唯一 Required | blocked | healthy | not_generated |
| ST-005 | 合法执行但所有 Required Tool 失败 | failed | healthy/degraded | not_generated |
| ST-006 | Budget 正常耗尽、部分完成 | partial | healthy | limitations |
| ST-007 | Task 完成、Guard protocol failed | completed | degraded | guard_degraded |
| ST-008 | Required RAG 无证据、系统全部 healthy | partial | healthy | limitations |
| ST-009 | Late Result 在封存后满足剩余 Requirement | 原状态不变 | 原状态不变 | 原状态不变 |
| ST-010 | 中间 Node 直接写 overall_status | 验收失败 | 不适用 | 不适用 |

## 16. Canonical Hash / Migration 故障矩阵

| ID | 输入差异 | Expected Hash/行为 |
|---|---|---|
| HS-001 | Object key 顺序不同 | content_hash 相同 |
| HS-002 | Semantic Set 顺序不同 | hash 相同 |
| HS-003 | Ordered Execution Steps 顺序不同 | hash 不同 |
| HS-004 | Decimal `1.0` vs 规范等价值 | 按规范化策略相同 |
| HS-005 | Unicode composed/decomposed | NFC 后相同 |
| HS-006 | Schema Version 不同 | hash 不同 |
| HS-007 | Null vs omitted | 按字段明确策略断言，不允许未定义 |
| HS-008 | 同手机号不同 Tenant/Purpose HMAC | fingerprint 不同 |
| HS-009 | HMAC Key Rotation | 新数据用新 key_id，旧数据可按历史 key 验证 |
| HS-010 | v1 Contract 直接用 v2 安全 Schema Replay | 拒绝，需 Migration + New Run |

## 17. Reason Code / Audit 验收

| ID | 条件 | 预期 |
|---|---|---|
| AU-001 | 同一错误来自不同 Node | 使用同一稳定 Reason Code，而非自由同义词 |
| AU-002 | 多原因 | 保存完整 reason_codes + 唯一 primary_reason_code |
| AU-003 | Reason Code 废弃 | 旧 Code 含义不变，指向 deprecated_by |
| AU-004 | Run 完成 | Audit 可追溯 Floor/Contract/Scope/Strategy Hash |
| AU-005 | 日志检查 | 不含完整用户金融数据、文档正文、Secret、Raw Model Response |
| AU-006 | Replay | parent/replay 关系明确，reuse count 正确 |
| AU-007 | Late Result | 有独立事件，不改最终 Status |

## 18. SLO 与红线

### 18.1 绝对红线

以下同时按 `violation_count` 和 `violation_rate` 统计；任何一次事件立即告警：

```text
required_drop_count = 0
forbidden_execution_count = 0
silent_scope_expansion_count = 0
guard_false_validated_count = 0
duplicate_side_effect_count = 0
```

### 18.2 可降级指标

```text
router_protocol_degraded_rate
extractor_protocol_degraded_rate
strategy_reconcile_rate
contract_integrity_repair_rate
unknown_step_rate
task_status_disagreement_rate
```

每个指标 MUST 记录：

```text
measurement_window
request_count
eligible_request_count
event_count
rate
runtime_revision
schema_versions
```

“无请求所以 rate=0”不得等价于已证明安全；红线必须由故障注入和回归集主动覆盖。

## 19. 切换准入条件

控制面 v2 只有同时满足以下条件才可从 Shadow 切为执行：

1. 四份设计文档评审通过，无未定义组合。
2. Schema Validation 和 Canonical Hash 测试全部通过。
3. 本文所有 P0 Fault Injection 通过。
4. 红线 `violation_count=0`。
5. Shadow 无 Side Effect、无 Memory Write、无 Production State Mutation。
6. Router/Extractor 双故障时 Floor Required 仍保留。
7. Scope/Permission/Effect Closure 无绕过路径。
8. Status 三层与矩阵完全一致。
9. Guard Degraded 从未显示为 Validated。
10. Rollback 到现有 v3.1 控制面已经演练。

## 20. 验收输出格式

每次回归生成：

```text
control_plane_acceptance_<revision>.json
control_plane_acceptance_<revision>.md
```

报告至少包含：

- Runtime/Schema/Prompt/Contract Versions
- Test ID、输入 Fixture Hash
- Expected vs Actual Contract/Strategy/Status
- Reason Code Diff
- Side Effect/Memory Mutation Counters
- 红线计数和 SLO
- 失败项、可重试性、Owner Component
- 是否允许切换 Feature Flag

最终 Gate 只能输出：

```text
PASS_SHADOW
PASS_CANARY
FAIL_CONTRACT_INVARIANT
FAIL_SIDE_EFFECT_SAFETY
FAIL_STATUS_SEMANTICS
FAIL_OBSERVABILITY
```

禁止用“最终答案看起来不错”替代控制面验收。
