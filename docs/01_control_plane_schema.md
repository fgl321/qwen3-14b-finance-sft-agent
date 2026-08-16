# 01 控制面 Schema 设计

> 状态：冻结候选设计的规范性定义  
> 文档版本：`control-plane-schema-v1`  
> 适用范围：金融 Agent 控制面 v2 Shadow、切换与后续正式实现  
> 非目标：本文不修改现有业务代码，不定义新的 Agent 节点，不展开 Claim Provenance DAG

## 1. 规范用语与基本原则

本文中的关键词具有以下强度：

- **MUST / 必须**：不可违反的契约或安全要求。
- **MUST NOT / 禁止**：任何降级路径都不得绕过。
- **SHOULD / 应当**：默认实现；例外必须记录稳定 `reason_code`。
- **MAY / 可以**：不影响契约正确性的可选行为。

控制面固定回答三个相互独立的问题：

1. `SealedEffectiveContract`：用户和平台要求必须满足什么、禁止什么、允许在哪些资源范围内做。
2. `RuntimeCapabilitySnapshot`：当前运行环境实际上能提供什么能力。
3. `EffectiveExecutionStrategy`：在不突破 Contract、Policy、Permission、Scope、Availability 和 Budget 的前提下，本 Run 怎么执行。

以下概念禁止互相覆盖：

- Availability 不得降低 Requirement。
- Router 不得生成、降低或覆盖 Sealed Contract。
- Strategy 不得扩大 Scope 或放宽 Permission。
- Budget/Deadline 不得反向修改用户 Requirement。
- 任何中间节点不得直接设置最终 `overall_status`。

## 2. 标识、版本和基础类型

所有 ID MUST 是稳定、不可复用的字符串。时间持久化 MUST 使用 UTC；运行中 Duration/TTL MUST 使用当前进程的 monotonic clock。

```python
from decimal import Decimal
from typing import Literal

SchemaVersion = str
ContractVersion = int
RuntimeRevision = str

RequestId = str
RunId = str
TraceId = str
StepId = str
TaskId = str
CapabilityId = str
ToolName = str
ReasonCode = str

RequirementLevel = Literal["not_needed", "optional", "required"]
PermissionLevel = Literal["allowed", "forbidden"]

Authority = Literal[
    "system_policy",
    "api_policy",
    "user_explicit",
    "caller_explicit",
    "semantic_extractor",
    "router_strategy",
]

EnforcementStrength = Literal[
    "hard_policy",
    "explicit_constraint",
    "default",
    "inferred",
]

RuntimeCapabilityStatus = Literal[
    "available",
    "degraded",
    "unavailable",
    "unknown",
]
```

`Authority` 表示来源身份，`EnforcementStrength` 表示该来源的约束强度。UI/API 默认值不得仅因来自 API 就覆盖用户显式约束。

## 3. 显式约束与 Capability Constraint

### 3.1 ConstraintSource

```python
class ConstraintSource:
    constraint_id: str
    authority: Authority
    enforcement_strength: EnforcementStrength
    rule_id: str

    # 运行时内存可保留原文；普通持久日志不保存 source_span_text。
    source_start: int | None
    source_end: int | None
    source_hash: str | None
    redacted_preview: str | None
```

日志中的 `redacted_preview` SHOULD 默认关闭，只允许在限时调试模式下保存。Validation Error MUST 删除可能包含用户原始值的 `input` 和敏感 `ctx`。

### 3.2 CapabilityConstraint

```python
class CapabilityConstraint:
    capability: CapabilityId
    requirement: RequirementLevel
    permission: PermissionLevel
    source: ConstraintSource
    scope_ref: str | None
```

Requirement 与 Permission 是两个维度：

| 示例 | requirement | permission |
|---|---|---|
| 不需要联网 | `not_needed` | `allowed` |
| 可以联网补充 | `optional` | `allowed` |
| 必须联网核验 | `required` | `allowed` |
| 不要联网 | `not_needed` | `forbidden` |

同一 Authority 内同一 Capability 同时出现 `required + forbidden` MUST 生成显式 Constraint Conflict，不能简单选择 forbidden 或 required。

### 3.3 ExplicitRequirementFloor

```python
class ExplicitRequirementFloor:
    schema_version: SchemaVersion
    request_id: RequestId
    constraints: tuple[CapabilityConstraint, ...]
    extraction_status: Literal["completed"]
    parser_version: str
    canonical_hash: str
```

Requirement Floor 是 **Explicit Constraint Parser**，不是 Intent Router Lite：

- MUST 只捕获“必须使用文档”“必须给引用”“禁止联网”“必须使用工具”等显式控制约束。
- MUST NOT 从“帮我算房贷”“哪种方案划算”等自然语言业务意图推断能力。
- SHOULD 高精度、低召回；未捕获的隐含要求由 Semantic Extractor 负责。

## 4. Semantic Extractor 与 Preliminary Router 输出

### 4.1 SemanticRequirementContract

```python
class TaskRequirement:
    task_id: TaskId
    description: str
    required: bool
    capabilities: tuple[CapabilityId, ...]
    depends_on: tuple[TaskId, ...]
    evidence_tool_names: tuple[ToolName, ...]
    requires_citations: bool

class SemanticRequirementContract:
    schema_version: SchemaVersion
    request_id: RequestId
    constraints: tuple[CapabilityConstraint, ...]
    task_requirements: tuple[TaskRequirement, ...]
    confidence: Decimal
    ambiguities: tuple[str, ...]
    invocation_status: Literal[
        "success", "repaired", "protocol_failed", "service_failed", "timeout"
    ]
    canonical_hash: str
```

Semantic Extractor MUST 只消费当前用户消息和必要历史上下文，MUST NOT 消费 Router fallback 输出。

### 4.2 PreliminaryStrategy

```python
class PreliminaryStrategy:
    schema_version: SchemaVersion
    request_id: RequestId
    orchestration_mode: Literal[
        "direct", "rag", "tool", "hybrid", "clarify", "unsupported"
    ]
    proposed_capabilities: tuple[CapabilityId, ...]
    proposed_tasks: tuple[TaskRequirement, ...]
    confidence: Decimal
    invocation_status: Literal[
        "success", "repaired", "protocol_failed", "service_failed", "timeout"
    ]
    canonical_hash: str
```

Preliminary Router SHOULD 同时接收用户请求和已经立即生成的 Floor，以提高策略正确率。Router 输出仅是执行建议，MUST NOT 成为 Requirement Contract 的 source of truth。

Floor 完成后，Extractor 与 Router SHOULD 并行执行，分别受独立子预算与共同 Request Deadline 约束。

## 5. Contract 生命周期

固定生命周期：

```text
ExplicitRequirementFloor
        +
SemanticRequirementContract
        ↓
MergedContractDraft
        ↓
IntegrityCheck / ContractRevisionRecord
        ↓
SealedEffectiveContract
        ↓
Execution
```

### 5.1 MergedContractDraft

```python
class ConstraintConflict:
    conflict_id: str
    capability: CapabilityId
    constraint_ids: tuple[str, ...]
    conflict_type: Literal[
        "same_authority_required_forbidden",
        "higher_authority_policy_block",
        "scope_conflict",
        "unclassified_constraint_conflict",
    ]
    user_resolvable: bool
    reason_code: ReasonCode

class MergedContractDraft:
    schema_version: SchemaVersion
    request_id: RequestId
    draft_version: int
    constraints: tuple[CapabilityConstraint, ...]
    task_requirements: tuple[TaskRequirement, ...]
    conflicts: tuple[ConstraintConflict, ...]
    parent_hashes: tuple[str, ...]
    canonical_hash: str
```

### 5.2 ContractRevisionRecord

```python
class ContractRevisionRecord:
    revision_id: str
    request_id: RequestId
    from_draft_hash: str
    to_draft_hash: str
    changes: tuple[dict, ...]
    reason_codes: tuple[ReasonCode, ...]
    repaired_at_utc: str
    integrity_gate_version: str
```

Integrity Gate MUST 只比较 Floor、Extractor 和 Draft 的结构化对象，MUST NOT 再扫描用户原文形成第二套 Parser。

### 5.3 SealedEffectiveContract

```python
class SealedEffectiveContract:
    schema_version: SchemaVersion
    contract_version: ContractVersion
    request_id: RequestId
    constraints: tuple[CapabilityConstraint, ...]
    task_requirements: tuple[TaskRequirement, ...]
    requested_scope_refs: tuple[str, ...]
    conflicts: tuple[ConstraintConflict, ...]
    sealed_at_utc: str
    canonical_hash: str
    parent_draft_hash: str
```

规则：

- Contract Draft 在 Seal 前 MAY 被 Integrity Gate 修复。
- Sealed Contract MUST 是不可变对象。
- Execution 开始后 MUST 始终满足 `current_contract_hash == sealed_contract_hash`。
- 执行中发现 Contract 错误时 MUST 停止、阻断或降级当前 Run；MUST NOT 原地热修改后继续。
- 用户澄清或安全字段迁移 MUST 创建 New Run + New Contract。
- 旧 Contract MUST 按创建时 Schema 解释，禁止安全字段 silent migration。

## 6. Resource Scope

### 6.1 RequestedResourceScope

```python
class RequestedResourceScope:
    scope_id: str
    source_constraint_ids: tuple[str, ...]
    requested_description: str
    allowed_source_types: tuple[str, ...]
    forbidden_source_types: tuple[str, ...]
    web_access: PermissionLevel
    freshness_requirement: str | None
    canonical_hash: str
```

### 6.2 ResolvedResourceRef 与 ResolvedResourceScope

```python
class ResolvedResourceRef:
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    document_version: int
    content_hash: str

class ResolvedResourceScope:
    schema_version: SchemaVersion
    scope_id: str
    requested_scope_hash: str
    resources: tuple[ResolvedResourceRef, ...]
    allowed_source_types: tuple[str, ...]
    forbidden_source_types: tuple[str, ...]
    web_access: PermissionLevel
    authorization_snapshot_id: str
    resolved_at_utc: str
    canonical_hash: str
    resolution_status: Literal[
        "resolved", "not_found", "unauthorized", "ambiguous", "failed"
    ]
```

Executor Preflight MUST 再验证：资源存在、版本与哈希一致、授权仍有效、目标未替换、Scope 未扩大。Scope 解析失败 MUST NOT 自动改用全知识库、其他同名文档或 Web。

## 7. Runtime Capability Snapshot

```python
class CapabilityAvailability:
    capability: CapabilityId
    provider_or_tool: str
    status: RuntimeCapabilityStatus
    checked_at_utc: str
    reason_code: ReasonCode | None
    retryable: bool
    estimated_recovery_seconds: int | None
    supported_degradations: tuple[str, ...]

class RuntimeCapabilitySnapshot:
    schema_version: SchemaVersion
    run_id: RunId
    observed_at_utc: str
    capabilities: tuple[CapabilityAvailability, ...]
    canonical_hash: str
```

Availability 只是规划时快照，不是 Contract，也不是执行结果。Executor 是最终权威；执行前或执行中能力状态变化 MUST 形成 Outcome，MUST NOT 修改 Requirement。

## 8. Tool Manifest、Effects 与权限闭包

### 8.1 Effect 等级

```python
NetworkEffect = Literal["none", "internal", "external", "unknown"]
DataEffect = Literal[
    "none", "local_read", "external_read", "external_write", "unknown"
]
MutationEffect = Literal["none", "reversible", "irreversible", "unknown"]
SensitiveDataEffect = Literal["none", "pii_read", "pii_egress", "unknown"]
CostEffect = Literal["free", "metered", "unknown"]

class ToolEffects:
    network: NetworkEffect
    data: DataEffect
    mutation: MutationEffect
    sensitive_data: SensitiveDataEffect
    cost: CostEffect
```

权限比较 SHOULD 使用固定 lattice；`unknown` MUST 按 `declared_max_effects` 的更严格风险处理。

### 8.2 ToolManifest

```python
class ToolManifest:
    tool_name: ToolName
    tool_version: str
    capability: CapabilityId
    declared_max_effects: ToolEffects
    supports_idempotency: bool
    supports_status_query: bool
    supports_compensation: bool
    deterministic: bool
    result_freshness_policy: str
```

### 8.3 ResolvedToolCall

```python
class ResolvedToolCall:
    step_id: StepId
    tool_name: ToolName
    tool_version: str
    normalized_arguments_hash: str
    resolved_scope_hash: str
    resolved_call_effects: ToolEffects
    effect_profile_hash: str
    provider_idempotency_key: str | None
```

Permission MUST 检查 Tool/Capability Effects 的传递闭包，而非只检查工具名称。“禁止联网”必须同时禁止所有 `network=external` 的间接工具调用。

## 9. Runtime Policy、Deadline 与 Budget

```python
class RuntimeBudget:
    max_llm_calls: int
    max_tool_calls: int
    max_retrieval_queries: int
    max_protocol_repairs: int
    max_execution_rounds: int
    token_budget: int | None
    monetary_budget: Decimal | None

class RuntimePolicyEnvelope:
    schema_version: SchemaVersion
    request_id: RequestId
    hard_limits: RuntimeBudget
    soft_targets: RuntimeBudget
    reserved_delivery_budget_ms: int
    created_at_utc: str
    deadline_at_utc: str
    original_budget_ms: int
    source_authority: Authority
    canonical_hash: str
```

运行时 MUST 使用当前进程 monotonic clock 创建 `monotonic_deadline`。Monotonic 值不得持久化或跨进程比较；进程重启后根据 UTC deadline 重新计算剩余时长并建立新的 monotonic deadline。

预算耗尽 MUST 保留 `required`，并产生 unmet Requirement 和稳定 Reason Code。预算正常生效通常不表示系统故障。

## 10. Effective Strategy 与 Reconciliation

```python
class StrategyStep:
    step_id: StepId
    capability: CapabilityId
    task_ids: tuple[TaskId, ...]
    depends_on: tuple[StepId, ...]
    tool_name: ToolName | None
    scope_hash: str | None
    expected_effects: ToolEffects | None
    required_outputs: tuple[str, ...]

class EffectiveExecutionStrategy:
    schema_version: SchemaVersion
    run_id: RunId
    sealed_contract_hash: str
    resolved_scope_hashes: tuple[str, ...]
    runtime_snapshot_hash: str
    preliminary_strategy_hash: str | None
    steps: tuple[StrategyStep, ...]
    strategy_status: Literal[
        "ready", "ready_degraded", "blocked", "unfulfillable"
    ]
    reason_codes: tuple[ReasonCode, ...]
    canonical_hash: str
```

Strategy Reconciler 是确定性 Validator + Repairer，只能修复 Strategy 遗漏，MUST 按以下顺序裁决：

```text
Authority / Policy
→ Permission
→ Scope
→ Requirement
→ Availability
→ Strategy
```

它 MUST NOT 突破 Policy、Permission、Scope、Availability 或 Budget。只有 `required + allowed + scope_valid + available/degraded且满足Policy` 时才可补齐执行分支。

## 11. Step 状态、幂等与 Replay

```python
StepExecutionState = Literal[
    "pending",
    "running",
    "succeeded",
    "failed_retryable",
    "failed_terminal",
    "cancelled",
    "unknown",
]

class StepIdempotencyRecord:
    idempotency_key: str
    request_id: RequestId
    run_id: RunId
    sealed_contract_hash: str
    resolved_scope_hash: str
    step_id: StepId
    tool_name: ToolName
    tool_version: str
    normalized_arguments_hash: str
    effect_profile_hash: str
    state: StepExecutionState
    result_ref: str | None
    original_completed_at_utc: str | None
    freshness_validated: bool
    provider_idempotency_key: str | None
```

稳定幂等键 MUST 至少覆盖上述身份字段。复用前 MUST 同时验证 Contract、Scope、Tool Version、Arguments、Effects 和 Freshness。

外部副作用若没有 Provider Idempotency，系统 MUST NOT 宣称 exactly-once。`unknown` 只能通过 Provider Status Query/Reconciliation 进入 `succeeded`、`failed_terminal` 或继续 `unknown`，禁止通过重复原操作解决。

## 12. Cancellation 与 Late Result

```python
class CancellationState:
    requested: bool
    cancel_requested_at_utc: str | None
    execution_cutoff_at_utc: str | None
    reason_code: ReasonCode | None

class LateResultRecord:
    step_id: StepId
    received_at_utc: str
    result_ref: str
    side_effect_state: StepExecutionState
    eligible_for_future_reuse: bool
```

取消或 Deadline 到达后：

- MUST 停止新的 Planner、RAG、Tool 和 Execution Round。
- SHOULD 协作式取消仍可取消的异步任务。
- 已成功结果不得丢失。
- Run 最终状态封存后，Late Result MUST 进入审计但不得修改已封存状态。
- Cancellation 不等于 Compensation；外部副作用可能已经发生。

## 13. Capability Outcome、三层状态与 Delivery

```python
CapabilityOutcomeStatus = Literal[
    "satisfied", "partially_satisfied", "unsatisfied", "not_required"
]

class CapabilityOutcome:
    capability: CapabilityId
    required: bool
    outcome: CapabilityOutcomeStatus
    actual_output_refs: tuple[str, ...]
    runtime_status: RuntimeCapabilityStatus
    allowed_degradation_used: str | None
    reason_codes: tuple[ReasonCode, ...]

TaskStatus = Literal["completed", "partial", "failed", "blocked"]
SystemHealth = Literal["healthy", "degraded", "failed"]
DeliveryStatus = Literal[
    "validated",
    "validated_with_limitations",
    "guard_degraded",
    "rejected",
    "not_generated",
]

class FinalRunStatus:
    task_status: TaskStatus
    system_health: SystemHealth
    delivery_status: DeliveryStatus
    degraded_components: tuple[str, ...]
    primary_reason_code: ReasonCode | None
    reason_codes: tuple[ReasonCode, ...]
    legacy_overall_status: str
```

`legacy_overall_status` 只允许由最终 Status Assembler 派生，用于旧前端兼容；任何节点禁止直接写入。

## 14. Capability Satisfaction Policy

```python
class CapabilitySatisfactionPolicy:
    capability: CapabilityId
    policy_version: str
    acceptable_runtime_statuses: tuple[RuntimeCapabilityStatus, ...]
    required_outputs: tuple[str, ...]
    minimum_quality: dict[str, str | int | Decimal]
    allowed_degradations: tuple[str, ...]
```

Capability 是否满足 MUST 根据实际产物和该 Policy 判断，不能仅根据组件 `healthy/degraded` 判断。

示例：Reranker 失败但混合召回与可验证引用完整，可形成 `task=completed + health=degraded`；Required Citation Validator 不可用且无法完成引用验证，则对应 Capability 未满足。

## 15. Reason Code Registry

```python
class ReasonCodeDefinition:
    code: ReasonCode
    category: Literal[
        "contract", "policy", "scope", "availability", "strategy",
        "execution", "retrieval", "delivery", "deadline", "budget", "audit"
    ]
    default_severity: Literal["info", "warning", "error", "critical"]
    retryable: bool
    task_impact: str
    health_impact: str
    delivery_impact: str
    owner_component: str
    introduced_version: str
    deprecated_by: ReasonCode | None
```

首版稳定 Reason Code 至少包括：

```text
CONTRACT_REQUIRED_CAPABILITY_MISSING
CONTRACT_PERMISSION_CONFLICT
CONTRACT_BLOCKED_BY_POLICY
CONTRACT_INTEGRITY_REPAIRED
CONTRACT_SEALED_MUTATION_DETECTED
CONTRACT_SCHEMA_MIGRATION_REQUIRED
CONTRACT_REQUIRED_FIELD_MISSING
SCOPE_RESOLUTION_FAILED
SCOPE_EXECUTION_PRECONDITION_FAILED
CAPABILITY_UNAVAILABLE
CAPABILITY_DEGRADED
STRATEGY_RECONCILED
STRATEGY_UNCLASSIFIED_CONTROL_STATE
TOOL_EXECUTION_FAILED
TOOL_RESULT_UNKNOWN
RETRIEVAL_NO_EVIDENCE
ROUTER_PROTOCOL_DEGRADED
EXTRACTOR_PROTOCOL_DEGRADED
GUARD_PROTOCOL_DEGRADED
DEADLINE_EXCEEDED
BUDGET_EXHAUSTED
RUN_CANCELLED
LATE_RESULT_RECORDED
```

Reason Code 发布后禁止复用或静默改变含义；需要变化时废弃旧 Code 并新增。

## 16. Run Audit Envelope

```python
class RunAudit:
    schema_version: SchemaVersion
    request_id: RequestId
    run_id: RunId
    trace_id: TraceId
    parent_run_id: RunId | None
    replay_of_run_id: RunId | None

    floor_contract_hash: str
    extractor_contract_hash: str | None
    sealed_contract_hash: str
    resolved_scope_hashes: tuple[str, ...]
    runtime_snapshot_hash: str
    strategy_hash: str

    runtime_revision: RuntimeRevision
    schema_versions: dict[str, str]

    started_at_utc: str
    sealed_at_utc: str
    execution_started_at_utc: str | None
    completed_at_utc: str | None
    deadline_at_utc: str
    cancellation_state: CancellationState

    task_status: TaskStatus
    system_health: SystemHealth
    delivery_status: DeliveryStatus
    degraded_components: tuple[str, ...]
    primary_reason_code: ReasonCode | None
    reason_codes: tuple[ReasonCode, ...]

    budget_allocated: dict[str, str | int]
    budget_consumed: dict[str, str | int]
    idempotency_reuse_count: int
    late_result_count: int
```

RunAudit MUST 追加写入事件或引用详细 Trace，不得保存完整用户消息、文档正文、Secret 或模型原始敏感输出。

## 17. Canonical Serialization、Hash 与 HMAC

所有 Contract Seal、Cache、Replay、Shadow Diff、Plan Signature 和 Tool Reuse MUST 使用同一 Canonical Serialization 规范：

- UTF-8、Unicode NFC。
- Object key 稳定排序。
- UTC 时间固定格式。
- Decimal 使用规范化十进制字符串，禁止二进制 Float 直接参与身份哈希。
- 明确定义 omitted/default/null 策略。
- Schema Version 必须进入 Canonical Payload。
- 有序列表保持顺序；仅标记为 Semantic Set 的列表允许排序去重。

用途分离：

```text
content_hash
→ SHA-256(canonical_payload)，用于完整性和对象身份。

identity_fingerprint
→ HMAC-SHA256(key, domain + tenant + field_type + purpose + normalized_value)，
  用于低熵敏感字段关联。
```

HMAC 记录 MUST 包含 `algorithm`、`key_id`、`purpose`。不同租户、字段和用途必须 Domain Separation；密钥支持轮换且不得进入 Audit。

## 18. Schema Migration

- Sealed Contract MUST 按创建时 Schema Version 解释。
- 旧 Run 禁止自动用最新 Schema 重新解释后 Replay。
- 安全相关的 Permission、Scope、Effects 字段禁止 Silent Migration。
- Migration 必须显式产生：`migration_id`、`from_version`、`to_version`、`migration_hash`。
- 迁移结果是 New Contract Draft，并创建 New Run；不得修改旧 Run。

## 19. Shadow Envelope

```python
class ShadowControlPlaneResult:
    request_id: RequestId
    production_run_id: RunId
    shadow_revision: str
    shadow_floor_hash: str
    shadow_extractor_hash: str | None
    shadow_contract_hash: str | None
    shadow_strategy_hash: str | None
    status_prediction: FinalRunStatus | None
    diff_reason_codes: tuple[ReasonCode, ...]
    side_effects_permitted: Literal[False]
    persisted_to_user_memory: Literal[False]
```

Shadow Runtime 只开放 LLM reasoning、Contract computation、Strategy generation 和 Static Validation；不得持有真实副作用 Capability Registry，不执行真实 Tool，不写长期记忆，不影响 Production 状态或最终回答。

## 20. 本 Schema 的硬性不变量

1. Required 不得被 Router、fallback 或 default 降级。
2. Forbidden 能力不得进入 Effective Strategy。
3. Scope 不得被静默扩大、替换或跨租户解析。
4. Availability 和 Budget 不得修改 Requirement Contract。
5. Strategy 不得声称完成未实际执行的 Capability。
6. Task Status 只由 Requirement Fulfillment 事实决定。
7. System Health 只由运行组件健康事实决定。
8. Delivery Status 只由交付验证事实与风险策略决定。
9. 中间节点不得写 `overall_status`。
10. Sealed Contract 在 Execution 开始后不可修改。
11. Reconciler 不得突破 Policy、Permission、Scope、Availability 或 Budget。
12. Forbidden 必须作用于 Capability Effects 传递闭包。
13. 无 Provider 幂等保证时不得宣称外部副作用 exactly-once。
14. `unknown` 副作用步骤不得自动重复原操作。
15. Run 封存后 Late Result 不得修改最终状态。
16. 结果复用必须验证 Contract、Scope、Tool Version、Effects 与 Freshness。
17. 身份、缓存与 Replay 使用统一 Canonical Serialization。
18. Budget Exhaustion 不得被误记为 Requirement 降级。
19. Reason Code 含义发布后不可复用。
20. Shadow、Replay、Cache Hit 不得重复产生业务副作用。
21. `unknown` 只能通过状态确认或人工 Reconciliation 解决。
22. Sealed Contract 按创建时 Schema 解释，安全字段禁止静默迁移。
23. UTC 用于持久化；Duration、TTL、Deadline 在进程内使用 monotonic clock。

## 21. 与现有实现的兼容边界

本设计是后续控制面 v2 的目标 Schema，不表示现有 `SemanticRouteDecision`、`RequestRequirementContract`、`ProductionFinanceGraphState` 已满足本文全部字段。

实施 MUST 先进入 Shadow：

```text
legacy_control_plane.execute = true
new_control_plane.execute = false
```

旧字段 `overall_status`、现有 Route 和 Completion Contract 在切换期可保留，但只能作为兼容映射或 Production Baseline，不能成为新控制面的 source of truth。
