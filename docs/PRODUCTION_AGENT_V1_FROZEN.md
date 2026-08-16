# Production Agent V1 — Final Frozen

Status: `PRODUCTION_AGENT_V1_FINAL_FROZEN`

核心不变量清单见 [PRODUCTION_AGENT_V1_INVARIANTS.md](./PRODUCTION_AGENT_V1_INVARIANTS.md)。

后续默认不再改主执行链，只允许模块内部 Bug 修复、性能优化、
Token/Latency 优化、新增 Capability、新增 Artifact Type、
评测/监控/可观测性增强。

## 三条铁律

```text
LLM Output ≠ System State
No Observation → No Execution Claim
No Commit Receipt → No Success Claim
```

最终交付原则：

```text
Final Response ⊆ Committed + Verified Reality
```

## 职责边界

```text
LLM 决定：WHAT / WHY / REFERENCE / NEXT
Python 决定：VALID / SAFE / AUTHORIZED / STATE / EXECUTE / VERIFY / COMMIT
```

## 唯一主链（51 步）

Request Boundary → Idempotency → Policy Snapshot → 加载已提交
Conversation/Task/Raw/Narrative/Capability Context → Resource/Result/
Capability Catalog → Context Budget → Semantic Router → Schema Validation
→ Consistency Validation → Semantic Repair/Retry → Reference Resolution →
Typed Contract Resolution → Conflict Detection → Mutation Canonicalization →
StateMutationIntent → Working State Patch → State Mutation Validation →
Fact/Memory Reconciliation → EffectiveTaskContract → Orchestration Decision
(STATE_UPDATE_ONLY / DIRECT / CATALOG_DIRECT / CLARIFY / PLAN) →
Planner(仅 PLAN) → Plan/Permission/Source Gate → Authorized Execution →
Capability Observation → Replan → Completion Contract → Synthesis Context →
Answer Synthesis Proposal → Deterministic Artifact Verification → Grounding
Binding → Artifact Materialization → Referential Integrity → State Claim
Binding → Source-Aware Output Guard → TurnCommitBundle → Version CAS →
Atomic Critical Commit → StateMutationReceipt → TurnCommitReceipt → Final
Response Construction → Delivery → Raw Transcript Append → LTM Promotion
Gate → Narrative Compaction → Trace/Metrics/Audit。

## 42 条冻结 Invariants

1. LLM 输出永远只是 Proposal。
2. Python 是系统状态唯一权威。
3. Policy 高于 LLM。
4. LLM 不能创建正式 Handle。
5. CAN / NEED / RUN 严格分域。
6. 所有跨域转换必须通过 Resolver。
7. Required + Forbidden 必须显式 Conflict。
8. Semantic Proposal 必须 Schema Valid。
9. Semantic Proposal 必须 internally consistent。
10. Semantic Repair 失败必须 Fail Closed。
11. Python fallback 禁止猜用户语义。
12. Same-turn duplicate mutation 必须 canonicalize。
13. 初次 Fact 写入不得产生 superseded。
14. Same-value Fact 写入必须 idempotent no-op。
15. Fact 替换才产生 superseded。
16. 所有状态变化只走 StateMutationIntent。
17. Working State ≠ Committed State。
18. No Receipt → No Mutation Success Claim。
19. state_update_only → Planner=0。
20. direct → Planner=0。
21. Required Capability ≠ Planner Required。
22. Task State ≠ LTM。
23. Raw Transcript ≠ Task State。
24. Narrative ≠ Current Truth。
25. LTM 不能覆盖明确 Task Fact。
26. not_needed 不能记成 failed。
27. empty 必须代表真实执行成功但结果为空。
28. Technical Failure ≠ Insufficient Evidence。
29. Calculation satisfied via derivation 必须绑定 verified CALC。
30. Unsupported CALC 绝不能满足 Calculation Capability。
31. 重要 Claim 必须 Grounded。
32. Forbidden Source 不存在善意例外。
33. 只有 Committed Artifact 可引用。
34. Focus 只能指向 Committed Artifact。
35. Current-turn mutation claim 需要 current receipt。
36. Committed-state reference 不需要 current receipt。
37. Guard 失败不能修改 Execution Truth。
38. Delivery 失败不能伪造 awaiting_information。
39. Commit 前必须 Referential Integrity PASS。
40. Critical State 要么全 Commit，要么保持旧状态。
41. Same request_id 不能重复业务执行。
42. Same-thread 并发必须经过 version CAS。

## Fact Scope

```text
turn     → 仅本轮，不继承
task     → 当前任务，默认不继承
session  → 稳定个人事实（年龄、年收入等），可继承
durable  → 长期记忆候选，经 Memory Authority 决定加载
```

写入规则：

```text
None → value            = create（无 superseded）
value → 同 value        = idempotent no-op（无 superseded）
旧 value → 新 value     = old superseded + new active
```

## Truth Hierarchy

```text
Policy Snapshot              = Safety Truth
Committed Task State         = Current Task Truth
Raw Transcript               = Historical Record Truth
Capability Observation       = Execution Truth
Evidence Observation         = Evidence Truth
Verified Artifact            = Result Truth
StateMutationReceipt         = Mutation Truth
TurnCommitReceipt            = Persistence Truth
LLM                          = 不是 Truth Source
```

## 三扇门

```text
Gate 1 Semantic Validity    系统真的理解用户要什么了吗？
Gate 2 Execution Truth      该计算/检索/Memory/修改真的发生了吗？
Gate 3 Commit Truth         真实状态真的持久化成功了吗？
```

任一 Gate 不通过，禁止声称对应成功。

## 本轮收口记录（2026-08-16）

- 全部返回分支接入 `turn_commit_receipt` + `mutation_receipt`，
  `commit_observability.after` 版本一致。
- Redis Lua 改为版本号 CAS，兼容旧 key 与首次提交。
- Guard 新增 `unverified_state_mutation_claim`，并允许
  committed-state reference（已提交事实引用）。
- 语义路由：记忆类请求产出 `fact_updates/extracted_facts` +
  `state_update_only`；`memory_read` 视为 Context 能力而非可执行能力；
  纯状态更新允许空 capabilities/task_requirements。
- Fact Scope 落地：session/durable 跨任务继承，turn/task 不继承。
- Mutation Canonicalization：同字段重复写入 last-write-wins；
  同值写入幂等 no-op；替换才产生 superseded。

## 二轮收口（2026-08-16，P0/P1）

### P0 — Verified Derivation Chain

顺序冻结为：

```text
Calculation Proposal
→ Canonical Operation Resolver（ADD/SUBTRACT/MULTIPLY/DIVIDE/
  PERCENT/MIN/MAX）
→ Python Deterministic Engine
→ Verified CALC Artifact
→ Capability Observation
→ Completion
→ Synthesis
```

新增运行时 invariant：

```text
financial_calculation.status == satisfied
AND satisfaction_source == derivation
⇒ result_refs != []
AND every referenced CALC.verification_status == verified
AND CALC.output != null
```

不满足直接报 invariant violation。Python 在本轮先 materialize 并验证
CALC，再把 verified handle 绑定进 `used_derivation_ids`；引用上一轮
已验证 CALC（如 RESULT_2.CALC_1）同样可满足派生型计算能力。

### P1 — ResolveFactScope

```text
mutation 显式提供 scope      → 使用新 scope
修改已有 fact 且 scope=null  → 继承 existing active fact.scope
首次创建且带 scope           → 使用 Semantic Proposal scope
首次创建且 scope=null        → 安全默认 scope=task
```

新增 invariant：

```text
Replace Fact AND mutation.scope is null ⇒ new.scope == old.scope
Scope widening（task→session/durable，session→durable）
必须来自显式 typed scope mutation
```

验证基线：远程全量 pytest `670 passed`；五轮核心链 E2E
（记录90/20 → 70万 verified CALC_1 → RESULT_2.CALC_1 解释 →
20→25 保留 task scope → 65万 verified CALC_2）全部 PASS，
版本链 [1,2,3,4,5]，Planner=0，Guard=pass。

## 三轮收口（2026-08-16，State Claim / Artifact / Status）

### State Claim Binding（类型化，不再扩关键词）

Synthesis 必须为状态类句子输出 `state_claim_bindings`：

```text
current_turn_mutation_ack
  → 校验本轮 has_mutation_intent + 字段在本轮 mutation fields

committed_state_reference
  → 校验 fact_refs 属于当前已提交 canonical facts
```

Guard 以 typed binding 为主路径；词法检测只在没有任何 binding 时作为
保守回退。禁止继续为“已修改/已更新/已保存”逐词加例外。

### Artifact Commit Boundary

```text
Capability Observation: materialized_artifact_refs（执行层临时 Artifact）
Commit 之后:              committed_result_refs（RESULT_n.CALC_n）
```

只有 committed Artifact 进入 `recent_results / result catalog /
next-turn references`。Guard 失败时 Artifact 可以不 Commit，但必须
明确标注为执行层 Materialized，而不是 Committed。

### Execution / Delivery / Task Status Separation

```text
execution_status   来自 capability/missing（Guard 失败不得改）
fulfillment_status 来自 missing_requirements
delivery_status    来自 Guard/fallback
overall_status     组合
task.status        = completed，除非真实 missing/clarification
```

`awaiting_information` 只能由真实 missing requirement /
`needs_clarification` 产生。

验证基线：远程全量 pytest `674 passed`；五轮核心链 E2E 新增断言
全部 PASS：T5 `execution_status=success`、
`fulfillment_status=fulfilled`、`delivery_status=completed`、
`task.status=completed`（非 awaiting_information）、
`committed_result_refs=["RESULT_5.CALC_2"]`、Guard=pass。

## 四轮收口（2026-08-16，Turn/Task 分离）

核心原则新增：

```text
Every Task is a Turn；但不是 Every Turn 都是 Task。
LLM Proposal ≠ Reality；Turn ≠ Task。
```

### Task Admission Gate（Python 结构事实）

从已验证 typed proposal 计算四个结构事实：

```text
has_semantic_mutation
has_task_goal
has_execution_requirement
has_resolved_task_reference
```

全部为假 → conversational turn（不建 TASK、不产生业务状态修改、
不触发 Mutation ACK）；有语义变更 → state_mutation；
有已解析任务引用 → existing_task；否则 new_task。

Conversational Direct：Task=0、Planner=0、RAG=0、Tool=0、
Business State Mutation=0；仍允许轻量 Synthesis 回复。

### 新增 10 条 Turn/Task Invariants

1. Every Task is a Turn; not every Turn is a Task.
2. state_update_only=true ⇒ at least one semantic mutation exists.
3. System bookkeeping 不得计入 semantic mutation。
4. No task semantics ⇒ no task handle allocation。
5. new_task + null goal + no mutation + no capability ⇒ invalid task proposal。
6. goal="unspecified task" 不得作为可提交生产 Task。
7. Empty MutationReceipt 不得触发 Mutation ACK。
8. Conversational Direct 不得创建 Task / Result Artifact。
9. Conversational Direct 默认 Planner/RAG/Tool 为 0。
10. LLM routing variance 不得改变安全或状态执行边界。

### 问候行为基线（你好）

```text
TASK allocation       0
Business state change 0
Mutation ACK           0
Planner                0
RAG                    0
Tool                   0
```

如果 Router 输出 `state_update_only=true + 无 mutation`，
Semantic Consistency 直接判 `STATE_UPDATE_ONLY_WITHOUT_MUTATION`
并进入 Repair / Fail Closed；ACK Renderer 在无真实语义变更时禁止调用，
永远不可能再输出“已更新状态，无事实或约束变更”。

验证基线：远程全量 pytest `680 passed`；E2E“你好”满足上述
全部 0 项 + 正常问候回复；五轮核心链继续全部 PASS。

## 五轮收口（2026-08-16，Memory / Delivery / Reference）

- Memory Load Gate：LTM 读取移到 Semantic Route 之后，由
  `memory_policy` 决定是否物理调用 `list_facts`；
  forbidden/not_needed 时 `long_memory_attempted=false`、
  `long_memory_loaded=0`，不再“先读 LTM 再丢弃”。
- Delivery Truth Conflict：`technical_failures` 进入 Synthesis
  Context 与 Output Guard；存在技术失败却声称“没有技术异常”
  → `DELIVERY_TRUTH_CONFLICT` → rewrite。
- Qualified Artifact Ref：Referential Integrity 拒绝裸 `.CALC_n`，
  只接受 `RESULT_n.CALC_n` 且必须已 Commit（测试锁定）。

验证基线：远程全量 pytest `689 passed`；“你好”与五轮核心链
E2E 继续全部 PASS。

## 部署 / 回滚

- 部署包：`deploy_receipt_20260816.tgz`（32 文件）+ manifest + apply 脚本。
- 最近部署：`receipt_20260816_203353`。
- 回滚备份：`/home/yjq/deploy_receipt_backup_20260816_203353`
  （更早版本保留在 `..._164552`、`..._164957`、`..._165127` 等目录）。
- 未创建 git commit。
