# Production Agent V1 — 冻结 Invariants

状态：`PRODUCTION_AGENT_V1_FINAL_FROZEN`

所有 Bug 只按层定位：
Semantic / Consistency / Task Admission / Mutation / Evidence Contract /
Capability Observation / Grounding / Guard / Commit / Concurrency。
禁止为单个句子、单个文档、单个字段加 Python 关键词补丁。

## 五条最高原则

```text
LLM Proposal ≠ Reality
Turn ≠ Task
No Observation → No Execution Claim
No Verified Evidence → No Evidence Claim
No Commit Receipt → No Success Claim
```

## 25 条核心 Invariants

1. Every Task is a Turn; not every Turn is a Task.
2. LLM Output is Proposal, not Reality.
3. No Task Semantics → No Task Handle.
4. state_update_only → effective semantic mutation exists.
5. No Mutation Receipt → No Mutation ACK.
6. Working State ≠ Committed State.
7. Policy > User Constraints > LLM Proposal.
8. CAN / NEED / RUN are separate typed domains.
9. Required + Forbidden → explicit conflict.
10. Python fallback must not invent semantics.
11. Logical Requirement identity never disappears.
12. Physical query dedup must preserve requirement provenance.
13. Every required logical requirement gets one RequirementObservation.
14. not_observed ≠ insufficient_evidence.
15. insufficient_evidence ≠ technical_unavailable.
16. Retrieval failure must never be reported as document absence.
17. Forbidden source cannot be legalized by disclosure.
18. financial_calculation=satisfied → verified CALC exists.
19. No Observation → No Execution Claim.
20. Only committed artifacts are referenceable.
21. Guard failure does not rewrite Execution Truth.
22. Delivery failure does not automatically mean awaiting_information.
23. No successful Commit → No success claim.
24. Same request_id → no duplicate business execution.
25. Same-thread mutation → version CAS.

最终回答：

```text
Final Response ⊆ Authorized + Verified + Committed Reality
```

## 27 条最终核心 Invariants（最终冻结版）

1. LLM Output is Proposal, never Reality.
2. Semantic Contract invalid after repair → Fail Closed.
3. Python fallback cannot invent user semantics.
4. Every Task is a Turn, not every Turn is a Task.
5. No Task Semantics → No Task Allocation.
6. state_update_only → effective semantic mutation exists.
7. Forbidden Source → physical execution blocked.
8. Working State ≠ Committed State.
9. EffectiveTaskContract is the only runtime task contract.
10. Logical Requirement Universe comes only from EffectiveTaskContract.
11. Query Planner may merge queries, never logical requirements.
12. Every required logical requirement must have exactly one observation.
13. not_observed ≠ insufficient_evidence ≠ technical_unavailable.
14. Retrieved Chunk ≠ Verified Evidence.
15. No Observation → No Execution Claim.
16. financial_calculation=satisfied → verified CALC exists.
17. Forbidden general knowledge cannot escape as “通用建议”.
18. Only committed Artifacts are referenceable.
19. Qualified Artifact Ref must include valid RESULT handle.
20. Current-turn mutation claim requires current receipt.
21. Existing committed-state reference requires canonical state.
22. Guard failure cannot rewrite Execution Truth.
23. Delivery failure cannot automatically mean awaiting_information.
24. No Commit → No Success Claim.
25. Same request_id → No duplicated business execution.
26. Same-thread write → Version CAS.
27. Final Response ⊆ Authorized + Observed + Verified + Committed Reality.

可执行测试：`agent/tests/unit/test_frozen_invariants.py`
（覆盖 2/3、4/5、6、13、10/11/12、16、18/19、22/23 等核心不变量）。

## Turn/Task 新增 10 条

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

## Verified Derivation Chain（P0）

```text
financial_calculation.status == satisfied
AND satisfaction_source == derivation
⇒ result_refs != []
AND every referenced CALC.verification_status == verified
AND CALC.output != null
```

不满足 → 运行时 invariant violation。

## ResolveFactScope（P1）

```text
Replace Fact AND mutation.scope is null ⇒ new.scope == old.scope
Scope widening（task→session/durable，session→durable）
必须来自显式 typed scope mutation
```

## Evidence Truth 用户可见措辞

| 状态 | 用户允许看到的表述 |
| --- | --- |
| direct_support | 根据文档…… |
| partial_support | 文档部分支持…… |
| background_support | 文档提供背景，但不足以直接确认…… |
| insufficient_evidence | 当前上传文档无法确认…… |
| not_observed | 本次检索未覆盖该要求…… |
| technical_unavailable | 检索服务技术异常…… |
| conflict | 检索证据存在冲突…… |
| assessment_protocol_failed | 本轮证据验证未成功完成…… |

禁止把 not_observed 说成 insufficient_evidence，
禁止把 technical_unavailable 说成“文档中不存在”。

## Memory Load Gate（物理阻断）

```text
memory_policy=forbidden / not_needed
⇒ long_memory_attempted=false
  long_memory_loaded=0
  不调用 list_facts（物理阻断，不是“先读再丢弃”）

memory_policy=required / optional
⇒ 才允许真正读取 LTM
```

## Delivery Truth Conflict

`technical_failures`（evidence_assessment 协议失败 / 检索服务异常 /
记忆读取异常）必须进入 Synthesis Context，并传播到 Output Guard：

```text
存在 technical_failures
但回答声称“没有技术异常/不存在技术问题”
⇒ DELIVERY_TRUTH_CONFLICT → rewrite
```

## Qualified Artifact Ref

```text
禁止裸引用：.CALC_1 / CALC_1
只允许：RESULT_2.CALC_1（RESULT 必须存在、CALC 属于该 RESULT、已 Commit）
```

Referential Integrity Gate 对不符合格式的引用直接阻止 Commit。

## Legacy Topic Isolation

`document_topic_n` 永远只能是 **PhysicalQueryGroup**，不能成为
LogicalEvidenceRequirement。`_build_physical_queries` 的 legacy topic
扩展现在保留原始逻辑 requirement id，无逻辑要求时不再发明
`knowledge_lookup`；Requirement Universe 只能来自 EffectiveTaskContract。

## Claim → Requirement 绑定

文档结论必须经过：

```text
Claim → LogicalRequirementID → RequirementObservation → Citation
```

证据状态为 `not_observed / technical_unavailable /
assessment_protocol_failed` 的要求所绑定的引用禁止出现在最终回答中；
Guard 检测到使用这类引用 → `blocked_evidence_citation_used` → rewrite。

## Single CapabilityObservation

`capability_outcomes` 是最终唯一 Capability Observation；
Completion / overall_status 只读取它，禁止其他模块再推导一份
不同的 capability 状态。

## Evidence Requirement Integrity（硬门）

```text
requires_citations=true（检索型任务）
+ evidence_requirements=[]
⇒ EVIDENCE_CONTRACT_INCOMPLETE
⇒ 禁止 fulfilled/completed
```

不允许退化为 generic `knowledge_lookup`；Coverage Gate 先验证
`Evidence Required → Expected Universe > 0 → Every ID observed →
statuses legal`，任何一步 NO 都不能假 PASS。

## User-Asserted Document Claim 隔离

用户说“文档明确写了 X”只能是 `USER_ASSERTED_DOCUMENT_CLAIM`
（最多作为 Retrieval Hint）；没有
`RequirementObservation + Citation` 验证前，不得升级为
Document Fact（B 类）。

## 15 条最高层系统铁律

1. LLM Proposal ≠ Reality.
2. Python may validate semantics, but must not invent semantics.
3. Turn ≠ Task.
4. EffectiveTaskContract is the only runtime task truth.
5. ResolvedResourceScope is the only runtime resource truth.
6. User-asserted document content ≠ verified document evidence.
7. Required document output must produce LogicalEvidenceRequirement.
8. Logical Requirement Universe is immutable after contract build.
9. Query Planner may merge queries, never requirements.
10. Every expected requirement must have exactly one observation.
11. not_observed ≠ insufficient_evidence ≠ technical_unavailable.
12. One Capability → one authoritative CapabilityObservation.
13. Only verified and grounded artifacts may become referenceable.
14. No Commit Receipt → no success/state-change claim.
15. Final Response ⊆ Authorized + Observed + Verified + Committed Reality.

## 基线

- 远程全量 pytest：`702 passed`
- 五轮核心链 E2E：PASS（记录→70万→解释→改25万→65万）
- “你好”E2E：Task=0 / Planner=0 / RAG=0 / Tool=0 / Mutation ACK=0
- 最新部署：见 `PRODUCTION_AGENT_V1_FROZEN.md`
