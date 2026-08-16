from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agent_graph.conversation_state import (
    AuthorizedResourceRef,
    CapabilityDescriptor,
    ConstraintUpdate,
    ConversationState,
    FactUpdate,
    build_router_context_block,
)
from app.rag.rag_types import SourceAuthorityContract


Capability = Literal[
    "general_explanation",
    "knowledge_retrieval",
    "financial_calculation",
    "resource_catalog_read",
    "memory_read",
    "complex_reasoning",
    "citation_validation",
]
OrchestrationMode = Literal[
    "direct", "rag", "tool", "hybrid", "clarify", "unsupported"
]
RequirementLevel = Literal["not_needed", "optional", "required"]
CitationRequirement = Literal["not_needed", "preferred", "required"]
GroundingRequirement = Literal["none", "supplemental", "authoritative", "exclusive"]
RetrievalScope = Literal[
    "none",
    "selected_documents",
    "current_attachment",
    "uploaded_documents",
    "all_accessible_knowledge_base",
]
TaskKind = Literal["retrieval", "calculation", "reasoning", "synthesis", "validation"]
CapabilityConstraintValue = Literal[
    "not_needed", "optional", "required", "forbidden"
]
MemoryConstraint = Literal[
    "required", "optional", "forbidden", "not_needed"
]
ROUTE_SCHEMA_VERSION = "semantic-route-v3"
_VALID_TASK_ID_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

ConversationRelation = Literal[
    "new_task",
    "continuation",
    "follow_up",
    "refinement",
    "correction",
    "confirmation",
    "missing_information_response",
    "cancel_previous",
]


class TaskReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "none", "resolved", "ambiguous", "unresolved"
    ] = "none"
    reference_type: Literal[
        "active_task", "previous_task", "explicit"
    ] | None = None
    task_handle: str | None = Field(
        default=None, max_length=80
    )
    confidence: float = Field(default=0.0, ge=0, le=1)


class PendingActionResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "none", "confirmed", "rejected", "ambiguous"
    ] = "none"
    action_handle: str | None = Field(
        default=None, max_length=80
    )


class ResourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_type: str = Field(default="document", max_length=40)
    reference_form: Literal[
        "explicit",
        "deictic",
        "ordinal",
        "previous_focus",
        "previous_selection",
    ] = "explicit"
    selected_handles: list[str] = Field(
        default_factory=list, max_length=20
    )
    confidence: float = Field(default=0.0, ge=0, le=1)
    status: Literal[
        "resolved", "ambiguous", "unresolved"
    ] = "unresolved"


class ResultReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: str | None = Field(default=None, max_length=80)
    artifact_handle: str | None = Field(
        default=None, max_length=80
    )
    status: Literal[
        "resolved", "ambiguous", "unresolved"
    ] = "unresolved"
    confidence: float = Field(default=0.0, ge=0, le=1)


class RouterProposedAction(BaseModel):
    """Action the router wants to propose before executing (LLM decision)."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    proposed_by: Literal[
        "assistant", "planner", "system"
    ] = "system"


class DocumentReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str = Field(min_length=1, max_length=200)
    reference_type: Literal[
        "title", "alias", "filename", "document_id"
    ] = "title"
    # Semantic strength is decided by the router, not by Python regexes.
    strength: Literal[
        "required", "preferred", "mention_only"
    ] = "required"


class ResourceConstraintPayload(BaseModel):
    """Resource/source constraints, orthogonal to capabilities.

    Capability constraints answer whether retrieval is allowed at all;
    resource constraints answer which documents may be touched.
    """

    model_config = ConfigDict(extra="forbid")

    include_documents: list[DocumentReference] = Field(
        default_factory=list,
        max_length=20,
    )
    exclude_documents: list[DocumentReference] = Field(
        default_factory=list,
        max_length=20,
    )
    exclusive: bool = False


class RequestRequirementContract(BaseModel):
    """User-required outcomes, independent from orchestration success."""

    retrieval_requirement: RequirementLevel = "not_needed"
    citation_requirement: CitationRequirement = "not_needed"
    calculation_requirement: RequirementLevel = "not_needed"
    needs_exact_calculation: bool = False
    required_capabilities: list[Capability] = Field(default_factory=list)
    task_requirements: list["TaskRequirement"] = Field(default_factory=list)


class SemanticRouteProtocolError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        requirement_contract: RequestRequirementContract | None = None,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.requirement_contract = requirement_contract
        self.validation_errors = validation_errors or []
        self.schema_version = ROUTE_SCHEMA_VERSION


class TaskRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    description: str = Field(min_length=1, max_length=300)
    required: bool = True
    capabilities: list[Capability] = Field(min_length=1, max_length=6)
    evidence_tool_names: list[str] = Field(default_factory=list, max_length=8)
    # Omitted by older/degraded router responses for ordinary tasks.  The
    # route-level validator still requires an explicit citation-bearing task
    # whenever citation_requirement=required.
    requires_citations: bool = False
    task_kind: TaskKind = "reasoning"
    depends_on: list[str] = Field(default_factory=list, max_length=12)
    # 一个 task 内需要逐条交付的子要求（例如“授权条件”“查询权利”）。
    # 由确定性代码从 description 拆分填充，LLM 可省略。
    required_outputs: list[str] = Field(default_factory=list, max_length=8)
    # 一个检索 task 需要逐条获取的证据单元（例如“等待期规则”“责任免除”）。
    # 必须由模型生成；Python 只负责把每个单元解析到真实文档并逐条聚合。
    evidence_requirements: list[str] = Field(
        default_factory=list,
        max_length=12,
    )


class SemanticRouteDecision(BaseModel):
    """Model-owned semantic decision that deterministic code can enforce."""

    model_config = ConfigDict(extra="forbid")

    orchestration_mode: OrchestrationMode
    required_capabilities: list[Capability] = Field(
        default_factory=list, max_length=6
    )
    task_requirements: list[TaskRequirement] = Field(
        default_factory=list, max_length=12
    )
    # Typed semantic contract: capability vs resource are separate dimensions.
    capability_constraints: dict[str, CapabilityConstraintValue] = Field(
        default_factory=dict
    )
    resource_constraints: ResourceConstraintPayload = Field(
        default_factory=ResourceConstraintPayload
    )
    memory_constraint: MemoryConstraint = "not_needed"
    source_authority: SourceAuthorityContract = Field(
        default_factory=SourceAuthorityContract
    )
    conversation_relation: ConversationRelation = "new_task"
    resolved_goal: str | None = Field(
        default=None, max_length=500
    )
    task_reference: TaskReference = Field(
        default_factory=TaskReference
    )
    pending_action_resolution: PendingActionResolution = Field(
        default_factory=PendingActionResolution
    )
    resource_references: list[ResourceReference] = Field(
        default_factory=list, max_length=20
    )
    result_references: list[ResultReference] = Field(
        default_factory=list, max_length=8
    )
    fact_updates: list[FactUpdate] = Field(
        default_factory=list, max_length=12
    )
    extracted_facts: list[FactUpdate] = Field(
        default_factory=list, max_length=12
    )
    constraint_updates: list[ConstraintUpdate] = Field(
        default_factory=list, max_length=8
    )
    state_update_only: bool = False
    proposed_action: RouterProposedAction | None = None
    retrieval_requirement: RequirementLevel = "not_needed"
    citation_requirement: CitationRequirement = "not_needed"
    grounding_requirement: GroundingRequirement = "none"
    retrieval_scope: RetrievalScope = "none"
    # 文档意图分层：强规则只处理“明确命令”，模糊提及交给语义路由。
    resource_intent: Literal[
        "unspecified",
        "selected",
        "all_uploaded",
        "none",
        "named_document",
        "mention_only",
    ] = "unspecified"
    scope_strength: Literal[
        "explicit_required",
        "explicit_preferred",
        "semantic_inferred",
        "mention_only",
    ] = "semantic_inferred"
    requested_title: str | None = Field(
        default=None,
        max_length=120,
    )
    needs_exact_calculation: bool = False
    needs_latest_data: bool = False
    needs_clarification: bool = False
    risk_level: Literal["low", "medium", "high"] = "low"
    confidence: float = Field(ge=0, le=1)
    ambiguities: list[str] = Field(default_factory=list, max_length=8)
    reason_summary: str = Field(min_length=1, max_length=500)
    # Observability: who produced the semantics and whether Python repaired
    # only protocol-level fields.
    semantic_contract_source: Literal[
        "deepseek", "python_fallback", "legacy"
    ] = "deepseek"
    semantic_contract_status: Literal[
        "valid", "normalized", "degraded"
    ] = "valid"
    protocol_repaired: bool = False
    normalization_repairs: list[str] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_citation_tasks(cls, value):
        if not isinstance(value, dict) or value.get("citation_requirement") != "required":
            return value
        data = dict(value)
        tasks = [dict(item) for item in data.get("task_requirements") or []]
        if tasks and not any("requires_citations" in item for item in tasks):
            for item in tasks:
                capabilities = set(item.get("capabilities") or [])
                if "knowledge_retrieval" in capabilities:
                    item["requires_citations"] = True
        data["task_requirements"] = tasks
        return data

    @model_validator(mode="after")
    def validate_consistency(self) -> "SemanticRouteDecision":
        capabilities = set(self.required_capabilities)
        task_capabilities = {
            capability
            for task in self.task_requirements
            if task.required
            for capability in task.capabilities
        }
        if not capabilities.issubset(task_capabilities):
            missing = sorted(capabilities - task_capabilities)
            raise ValueError(f"required capabilities have no required task: {missing}")
        if self.citation_requirement != "not_needed":
            if self.retrieval_requirement == "not_needed":
                raise ValueError("citations require retrieval")
            if "knowledge_retrieval" not in capabilities:
                raise ValueError("citations require the knowledge_retrieval capability")
        if self.grounding_requirement in {"authoritative", "exclusive"}:
            if self.retrieval_requirement != "required":
                raise ValueError("authoritative/exclusive grounding requires retrieval")
        if self.retrieval_requirement == "required" and "knowledge_retrieval" not in capabilities:
            raise ValueError("required retrieval must be a required capability")
        if self.orchestration_mode == "hybrid":
            if not {"knowledge_retrieval", "financial_calculation"}.issubset(capabilities):
                raise ValueError("hybrid mode requires retrieval and calculation")
        if self.orchestration_mode == "clarify" and not self.needs_clarification:
            raise ValueError(
                "clarify mode requires needs_clarification=true"
            )
        if self.needs_clarification and self.orchestration_mode != "clarify":
            raise ValueError("needs_clarification=true requires clarify mode")
        if self.retrieval_requirement == "not_needed" and self.retrieval_scope != "none":
            raise ValueError("retrieval_scope must be none when retrieval is not needed")
        exact_financial_tasks = [
            task
            for task in self.task_requirements
            if task.required
            and task.task_kind == "calculation"
            and "financial_calculation" in task.capabilities
        ]
        if self.needs_exact_calculation and any(
            not task.evidence_tool_names for task in exact_financial_tasks
        ):
            raise ValueError(
                "every required exact-calculation task must declare evidence_tool_names"
            )
        if self.citation_requirement == "required" and not any(
            task.required and task.requires_citations
            for task in self.task_requirements
        ):
            raise ValueError(
                "required citations must be represented by a required task"
            )
        return self

    @model_validator(mode="after")
    def validate_resource_exclusive(self) -> "SemanticRouteDecision":
        if (
            self.resource_constraints.exclusive
            and not self.resource_constraints.include_documents
            and not self.requested_title
        ):
            raise ValueError(
                "exclusive resource scope requires include_documents"
            )
        return self


class SemanticRouterClient(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        thinking_enabled: bool = False,
        max_completion_tokens: int = 1600,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


_SYSTEM_PROMPT = """
You are the semantic router for a production financial agent. Do not answer the
user. Return exactly one JSON object and no Markdown or hidden reasoning.

Infer the user's deliverables, required capabilities, whether retrieval is a
necessary condition for a valid answer, citation strictness, and document
grounding strictness from the complete meaning of the request. Do not classify
from isolated keywords.

You are interpreting one turn inside an ongoing conversation. Never assume the
current utterance is standalone if relevant conversation context exists.
Before deciding intent, determine whether the user is:
1. starting a new task;
2. continuing the active task;
3. confirming a pending action;
4. supplying missing information;
5. refining or correcting the previous request;
6. referring to previous resources;
7. referring to previous results;
8. changing to an unrelated task.

When the current turn directly continues the scenario the user just set up
in the immediately preceding turns (for example the user records cash and
down_payment, then asks "支付首付后还剩多少钱"), classify it as
continuation/follow_up of the active task — do not open a new task, otherwise
the just-recorded task facts would be dropped.

Pure social turns (greeting, thanks, acknowledgment, goodbye — e.g. "你好",
"谢谢", "好的", "再见") with no request are conversational turns, not tasks:
set resolved_goal=null, task_requirements=[], required_capabilities=[] and
orchestration_mode=direct.  Do not invent a goal such as "问候用户".

Resolve references using the supplied conversation state, focused resources,
recent results, and the authorized resource catalog.  Never invent resource
handles: every handle must come from the authorized_resource_catalog.  If
multiple interpretations remain plausible after using all context, return
ambiguous.  Do not ask for clarification merely because the current sentence
is short or elliptical.

Ordinal, deictic, and contrastive references ("第二个", "这个", "这两个",
"前者", "后者", "另一个", "刚才那个") must be resolved FIRST against
conversation_state.focused_resources and conversation_state.recent_results
when those resources exist.  Only fall back to the structure of the previous
answer when no focused resource or previous result matches.  When a focused
resource matches, put its handle in resource_references.selected_handles with
reference_form=ordinal/deictic/previous_focus and status=resolved.
When focused_resources contains multiple resources, an ordinal like "第二个"
means the Nth resource in that list (in its listed order) — not a chapter or
section inside the first resource.  Only interpret the ordinal as an
intra-document section when the user explicitly mentions a section/chapter of
one specific document.

Field rules:
- orchestration_mode: direct/rag/tool/hybrid/clarify/unsupported
- required_capabilities: general_explanation/knowledge_retrieval/
  financial_calculation/resource_catalog_read/memory_read/complex_reasoning/
  citation_validation
- capability_constraints: typed capability map, values are
  not_needed/optional/required/forbidden.  This answers "may the system do X".
- resource_constraints: typed resource map, orthogonal to capabilities.
  include_documents/exclude_documents accept {reference, reference_type} where
  reference_type is title/alias/filename/document_id; exclusive=true means only
  the included documents may be used.  Each reference also carries strength:
  required (user explicitly demands it), preferred (nice to have), or
  mention_only (just a passing mention, must not lock or conflict scope).
  This answers "on which documents".
- memory_constraint: required/optional/forbidden/not_needed.  "不得使用模型记忆"
  / "不要用记忆补充" maps to forbidden; a self-contained question maps to
  not_needed; explicit memory continuation maps to required/optional.
- source_authority: typed contract with allowed/forbidden values for
  current_user_facts, selected_documents, deterministic_derivation, memory,
  general_model_knowledge, domain_heuristics, web.  Map explicit user
  constraints: "禁止使用模型记忆/不要用记忆补充" -> memory=forbidden;
  "禁止使用通用模型知识/不要凭常识/不要用常识" ->
  general_model_knowledge=forbidden; "禁止使用经验法则/不要用经验规则" ->
  domain_heuristics=forbidden; "禁止联网/不要联网" -> web=forbidden;
  "只允许使用X/仅依据X" -> selected_documents=allowed, and when the user
  requires the answer to come only from documents also set
  general_model_knowledge=forbidden and domain_heuristics=forbidden.
  Never relax a forbidden source on follow-ups unless the user explicitly
  allows that source in the current turn.
- conversation_relation:
  new_task/continuation/follow_up/refinement/correction/confirmation/
  missing_information_response/cancel_previous.  Decide from the full
  conversation, never from a keyword list.
- resolved_goal: one concise sentence stating what the user actually wants in
  this turn, after resolving references (e.g. "将此前年度必要支出转换为月度",
  "继续上一轮的文档对比任务").  null only when truly standalone.
- task_reference: {status: none/resolved/ambiguous/unresolved,
  reference_type: active_task/previous_task/explicit, task_handle,
  confidence}.  task_handle must come from conversation_state.active_task or a
  previously seen task handle; never invent task handles.
- pending_action_resolution: {status: none/confirmed/rejected/ambiguous,
  action_handle}.  Only set action_handle when the user clearly confirms or
  rejects the pending_action from conversation_state.
- resource_references: array of {resource_type, reference_form:
  explicit/deictic/ordinal/previous_focus/previous_selection,
  selected_handles, confidence, status: resolved/ambiguous/unresolved}.
  selected_handles must be handles from authorized_resource_catalog.
- result_references: array of {handle, status: resolved/ambiguous/unresolved,
  confidence}; handle must come from conversation_state.recent_results.
  When the user refers to a sub-result (e.g. "刚才存款保险那个结论",
  "70万那个计算", "第二个结论"), also fill artifact_handle with the exact
  sub-artifact handle inside that result (CLAIM_n / CALC_n / CONCLUSION_n).
  artifact_handle must come from the referenced result's claims/conclusions/
  calculations; never invent sub-artifact handles.
  When several recent_results exist, select the artifact whose
  type/summary/conclusions match the current question (e.g. "刚才的风险判断"
  -> the result whose conclusions cover platform risk), not merely the newest
  one.  Only pick the newest when it is actually the relevant one.  When the
  user asks to repeat/recall a previous conclusion ("再说一次", "刚才的结论",
  "关于X的结论"), resolve result_references to the artifact whose summary or
  claims cover that topic and set status=resolved; do not leave it empty when
  a matching artifact exists.
- fact_updates: array of {field, operation: set/replace/clear, value, source}.
  Fill when the current turn changes, corrects, or asks the system to remember
  a numeric/condition fact about the user (e.g. "首付款改成25万" ->
  operation=replace, value=250000; "记住我的房贷余额是80万" -> operation=set,
  field=mortgage_balance, value=800000).  Never invent fields; leave empty for
  pure questions that do not add or change user facts.
- extracted_facts: array of {field, operation: set, value, source} when the
  user supplies personal facts the system should store, including memory
  requests ("请记住/保存/记录我的年龄是35岁，家庭年收入36万元" ->
  field=age, value=35, and field=annual_income, value=360000).  Also fill when
  the user is starting a new task and states initial facts (e.g. "我有90万现金，
  首付款20万" -> cash=900000, down_payment=200000).  Leave empty for pure
  questions that only restate facts already stored.
  Prefer stable English field names such as age, annual_income,
  annual_expense, mortgage_balance, cash, down_payment, emergency_fund; use
  the same field name for the same fact across turns.
  Every fact carries a scope: turn (one-off assumption for this turn only),
  task (working parameter for the current task), session (stable personal
  facts such as age, annual income, family size that should survive across
  tasks), or durable (explicitly requested long-term storage).  Stable
  personal facts from memory requests use scope=session; explicit long-term
  save requests use scope=durable; one-off hypotheses use scope=turn; task
  parameters use scope=task.  Omit scope only when unsure; Python will fall
  back to task for first creation, and a replace without scope inherits the
  existing fact's scope (scope is never silently widened).
- constraint_updates: array of {name, value, operation: set/clear}.  Only fill
  when the user explicitly changes a constraint (e.g. web/memory/source rule).
  If the user does not modify constraints, keep empty so the previous task
  Source Authority is inherited unchanged.
- state_update_only: true when the current turn is purely a state change and
  no answer/execution is needed (e.g. "这一轮允许使用通用模型知识，但仍然禁止
  联网" or "请记住/保存/记录这些信息"): the system will ACK the update without
  running RAG/tools.  Memory requests that also expect a short confirmation
  reply are still state_update_only; the ACK is that confirmation.
- proposed_action: null by default. 只有当“执行某个动作前需要用户明确确认”时
  才填写 {action_type, description}（例如查询资源目录）；可直接执行的任务保持
  null，不要为了确认而确认。
- retrieval_requirement: not_needed/optional/required
- citation_requirement: not_needed/preferred/required
- grounding_requirement:
  none = the answer does not depend on documents;
  supplemental = documents can supplement general reasoning;
  authoritative = material conclusions need document evidence;
  exclusive = only the selected documents may support the answer.
- retrieval_scope: none/selected_documents/current_attachment/
  uploaded_documents/all_accessible_knowledge_base. Never invent document IDs.
- resource_intent:
  named_document = 用户把某份文档当作检索对象；
  mention_only = 只是提到文档名，并非要求检索它；
  selected/all_uploaded/none/unspecified = 依据请求语义。
- scope_strength:
  explicit_required = “必须/只根据”等明确命令 + 文档；
  explicit_preferred = “请根据/结合/提到”等较软表达；
  mention_only = 仅提及；
  semantic_inferred = 其他由语义推断。
- requested_title: 若用户点名《标题》，提取标题原文；否则 null。
- task_requirements: enumerate concrete deliverables and their capabilities.
- evidence_requirements: 对每个 retrieval task，列出必须逐条检索的证据单元
  （例如“等待期规则”“医院完整释义”“责任免除”“补偿原则”），
  不要把“计算案例B最终金额”当成检索要求；
  计算所需规则才属于证据单元。retrieval task 的 evidence_requirements
  必须是非空数组（至少 2 项），每个单元是一个独立的检索目标。
- task_kind separates evidence acquisition (retrieval), deterministic calculation,
  reasoning, final synthesis, and validation. A summary derived from several prior
  evidence/tool tasks is synthesis and must not create its own retrieval query.
- depends_on lists upstream task ids for derived reasoning/synthesis deliverables.
- For every exact-calculation deliverable, evidence_tool_names must name the
  deterministic business tool(s) whose successful outputs prove that specific
  deliverable complete. A unit conversion is only evidence for conversion, not
  for a downstream emergency-fund or insurance-gap deliverable.
- Set requires_citations=true for every deliverable whose conclusion must be
  supported by uploaded/selected document evidence.
- confidence: 0..1. Put material uncertainty in ambiguities. Use clarify only
  when a safe scope or deliverable truly cannot be selected.

Consistency rules: citation requirements imply retrieval; hybrid implies both
retrieval and calculation; every required capability must be used by at least
one required task. Authorization, tenant isolation, tool arguments, and runtime
budgets are enforced elsewhere and are not your decisions.

Capability vs resource: they are different dimensions and must never be mixed.
"不要检索任何文档/不要使用知识库/不要查资料" means knowledge_retrieval=forbidden.
"不要使用其他文档/不要使用别的文档/除A以外不要使用任何文档/只允许参考A/
仅根据A回答" means resource_constraints.exclusive=true with include_documents=[A]
and NEVER knowledge_retrieval=forbidden.  "不要使用A" means
resource_constraints.exclude_documents=[A], retrieval stays allowed.
"不要联网" only sets capability_constraints.web_search=forbidden and never
touches private knowledge retrieval.  "必须检索A" means
knowledge_retrieval=required and include_documents=[A].
""".strip()


class SemanticRouter:
    def __init__(
        self,
        *,
        llm_client: SemanticRouterClient,
        max_repairs: int = 1,
        max_completion_tokens: int = 2400,
    ) -> None:
        self.llm_client = llm_client
        self.max_repairs = max(0, int(max_repairs))
        self.max_completion_tokens = max(256, int(max_completion_tokens))
        self.last_requirement_contract: RequestRequirementContract | None = None

    async def route(
        self,
        user_message: str,
        *,
        tool_catalog: list[dict[str, str]] | None = None,
        conversation_state: ConversationState | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        resource_catalog: list[AuthorizedResourceRef] | None = None,
        capability_catalog: list[CapabilityDescriptor] | None = None,
        scope_snapshot: dict[str, Any] | None = None,
        narrative_segments: list[dict[str, Any]] | None = None,
    ) -> SemanticRouteDecision:
        catalog = list(tool_catalog or [])
        allowed_tool_names = {
            str(item.get("name") or "").strip()
            for item in catalog
            if str(item.get("name") or "").strip()
        }
        context_block = build_router_context_block(
            user_message=user_message,
            recent_messages=list(recent_messages or []),
            conversation_state=conversation_state,
            resource_catalog=list(resource_catalog or []),
            capability_catalog=list(capability_catalog or []),
            scope_snapshot=scope_snapshot,
            narrative_segments=narrative_segments,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{context_block}\n"
                    "<deterministic_tool_catalog>\n"
                    f"{json.dumps(catalog, ensure_ascii=False)}\n"
                    "</deterministic_tool_catalog>"
                ),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_repairs + 1):
            result = await self.llm_client.chat(
                messages=messages,
                thinking_enabled=False,
                max_completion_tokens=self.max_completion_tokens,
                response_format={"type": "json_object"},
            )
            raw = str((result.get("message") or {}).get("content") or "").strip()
            try:
                payload = self._normalize_payload(json.loads(raw))
                self.last_requirement_contract = self.requirement_contract_from_payload(
                    payload
                )
                decision = SemanticRouteDecision.model_validate(payload)
                consistency_violations = (
                    validate_semantic_consistency(decision)
                )
                if consistency_violations:
                    raise ValueError(
                        "semantic contract inconsistent: "
                        + ";".join(consistency_violations)
                    )
                if decision.normalization_repairs:
                    decision = decision.model_copy(
                        update={
                            "semantic_contract_status": "normalized"
                        }
                    )
                unknown_tools = sorted(
                    {
                        name
                        for task in decision.task_requirements
                        for name in task.evidence_tool_names
                        if name not in allowed_tool_names
                    }
                )
                if unknown_tools:
                    raise ValueError(
                        f"unknown evidence_tool_names: {unknown_tools}"
                    )
                return decision
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_repairs:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": raw[:4000]},
                        {
                            "role": "user",
                            "content": (
                                "The JSON failed schema/consistency validation: "
                                f"{type(exc).__name__}: {str(exc)[:500]}. "
                                "Preserve the intended semantics and return one corrected JSON object."
                            ),
                        },
                    ]
                )
        raise SemanticRouteProtocolError(
            "semantic router protocol failed: "
            f"{type(last_error).__name__}: {str(last_error)[:500]}",
            requirement_contract=self.last_requirement_contract,
            validation_errors=(
                last_error.errors(include_url=False)
                if isinstance(last_error, ValidationError)
                else []
            ),
        )
    @staticmethod
    def requirement_contract_from_payload(
        payload: dict[str, Any],
    ) -> RequestRequirementContract:
        tasks = [
            TaskRequirement.model_validate(task)
            for task in (payload.get("task_requirements") or [])
            if task.get("required", True)
        ]
        capabilities = list(
            dict.fromkeys(
                [
                    *list(payload.get("required_capabilities") or []),
                    *(capability for task in tasks for capability in task.capabilities),
                ]
            )
        )
        calculation_required = any(
            task.task_kind == "calculation"
            and "financial_calculation" in task.capabilities
            for task in tasks
        )
        return RequestRequirementContract(
            retrieval_requirement=payload.get("retrieval_requirement", "not_needed"),
            citation_requirement=payload.get("citation_requirement", "not_needed"),
            calculation_requirement="required" if calculation_required else "not_needed",
            needs_exact_calculation=bool(payload.get("needs_exact_calculation")),
            required_capabilities=capabilities,
            task_requirements=tasks,
        )

    @staticmethod
    def _normalize_payload(payload: Any) -> dict[str, Any]:
        """Normalize harmless router aliases before strict validation."""

        if not isinstance(payload, dict):
            raise TypeError("semantic route must be a JSON object")
        route_fields = set(SemanticRouteDecision.model_fields)
        normalized = {
            key: value for key, value in payload.items() if key in route_fields
        }
        normalization_repairs: list[str] = []
        capability_constraints = normalized.get("capability_constraints") or {}
        if not isinstance(capability_constraints, dict):
            capability_constraints = {}
        capability_constraints = {
            str(key).strip(): str(value).strip()
            for key, value in capability_constraints.items()
            if str(key).strip()
            and str(value).strip()
            in {"not_needed", "optional", "required", "forbidden"}
        }
        normalized["capability_constraints"] = capability_constraints

        memory_value = capability_constraints.get("memory_read")
        normalized["memory_constraint"] = {
            "required": "required",
            "optional": "optional",
            "forbidden": "forbidden",
        }.get(memory_value, "optional")
        raw_authority = normalized.get("source_authority") or {}
        authority = (
            dict(raw_authority)
            if isinstance(raw_authority, dict)
            else {}
        )
        if memory_value == "forbidden":
            authority["memory"] = "forbidden"
        if capability_constraints.get("web_search") == "forbidden":
            authority["web"] = "forbidden"
        normalized["source_authority"] = authority

        raw_resource = normalized.get("resource_constraints") or {}
        if not isinstance(raw_resource, dict):
            raw_resource = {}
        raw_include = raw_resource.get("include_documents") or []
        raw_exclude = raw_resource.get("exclude_documents") or []
        if any(isinstance(item, str) for item in raw_include) or any(
            isinstance(item, str) for item in raw_exclude
        ):
            normalization_repairs.append(
                "resource_references_normalized"
            )
        normalized["resource_constraints"] = {
            "include_documents": SemanticRouter._normalize_document_references(
                raw_include
            ),
            "exclude_documents": SemanticRouter._normalize_document_references(
                raw_exclude
            ),
            "exclusive": bool(raw_resource.get("exclusive", False)),
        }

        include_refs = normalized["resource_constraints"]["include_documents"]
        title_ref = next(
            (
                item
                for item in include_refs
                if item.get("reference_type") == "title"
            ),
            None,
        )
        if title_ref and not normalized.get("requested_title"):
            normalized["requested_title"] = str(title_ref["reference"])[:120]
        if (
            normalized["resource_constraints"]["exclusive"]
            and not include_refs
            and normalized.get("requested_title")
        ):
            normalized["resource_constraints"]["include_documents"] = [
                {
                    "reference": str(normalized["requested_title"]),
                    "reference_type": "title",
                }
            ]
        if (
            include_refs
            and normalized.get("resource_intent", "unspecified")
            == "unspecified"
        ):
            normalized["resource_intent"] = "named_document"
        if (
            normalized["resource_constraints"]["exclusive"]
            and normalized.get("scope_strength", "semantic_inferred")
            == "semantic_inferred"
        ):
            normalized["scope_strength"] = "explicit_preferred"

        raw_tasks = normalized.get("task_requirements") or []
        task_fields = set(TaskRequirement.model_fields)
        tasks: list[dict[str, Any]] = []
        for index, raw_task in enumerate(raw_tasks, start=1):
            if not isinstance(raw_task, dict):
                raise TypeError(f"task_requirements[{index}] must be an object")
            task = {
                key: value for key, value in raw_task.items() if key in task_fields
            }
            if not str(task.get("description") or "").strip():
                task["description"] = str(
                    raw_task.get("deliverable")
                    or raw_task.get("task")
                    or f"Complete required deliverable {index}"
                )
            task.setdefault("required", True)
            task.setdefault("evidence_tool_names", [])
            task.setdefault("requires_citations", False)
            if not task.get("task_kind"):
                capabilities = set(task.get("capabilities") or [])
                if task.get("evidence_tool_names"):
                    task["task_kind"] = "calculation"
                elif capabilities == {"citation_validation"}:
                    task["task_kind"] = "validation"
                elif "knowledge_retrieval" in capabilities:
                    task["task_kind"] = "retrieval"
                elif "complex_reasoning" in capabilities:
                    task["task_kind"] = "synthesis"
                else:
                    task["task_kind"] = "reasoning"
            elif task.get("task_kind") not in {
                "retrieval", "calculation", "reasoning", "synthesis", "validation"
            }:
                task["task_kind"] = "reasoning"
            raw_evidence = (
                task.get("evidence_requirements")
                or raw_task.get("evidence_queries")
                or raw_task.get("evidence_units")
                or raw_task.get("retrieval_requirements")
            )
            if isinstance(raw_evidence, str):
                raw_evidence = [raw_evidence]
            evidence_units = [
                str(item).strip()
                for item in (raw_evidence or [])
                if str(item).strip()
            ]
            if (
                not evidence_units
                and task.get("task_kind") == "retrieval"
            ):
                parts = [
                    part.strip()
                    for part in re.split(
                        r"[、；，,。]",
                        str(task.get("description") or ""),
                    )
                    if part.strip() and len(part.strip()) >= 2
                ]
                if len(parts) >= 2:
                    evidence_units = parts[:12]
            task["evidence_requirements"] = evidence_units
            depends_on = task.get("depends_on")
            if isinstance(depends_on, str):
                task["depends_on"] = [depends_on] if depends_on.strip() else []
            elif not isinstance(depends_on, list):
                task["depends_on"] = []
            tasks.append(task)
        raw_ids = [task.get("id") for task in tasks]
        tasks = SemanticRouter._normalize_task_ids(tasks)
        if any(
            task.get("id") != raw
            for task, raw in zip(tasks, raw_ids)
        ):
            normalization_repairs.append("task_ids_normalized")
        normalized["task_requirements"] = tasks
        ambiguities = normalized.get("ambiguities")
        if isinstance(ambiguities, str):
            normalized["ambiguities"] = [ambiguities] if ambiguities.strip() else []
        elif ambiguities is None:
            normalized["ambiguities"] = []
        if not str(normalized.get("reason_summary") or "").strip():
            normalized["reason_summary"] = (
                "Semantic route normalized from the model's structured intent, "
                "required capabilities, and task requirements."
            )
        if normalized.get("citation_requirement") == "required":
            retrieval_tasks = [
                task
                for task in tasks
                if task.get("required", True)
                and "knowledge_retrieval" in (task.get("capabilities") or [])
            ]
            if retrieval_tasks and not any(
                bool(task.get("requires_citations")) for task in tasks
            ):
                for task in retrieval_tasks:
                    task["requires_citations"] = True
        if capability_constraints:
            # Typed capability constraints are the semantic source of truth.
            # Legacy fields are derived so the rest of the pipeline keeps one
            # consistent view.
            retrieval_value = capability_constraints.get(
                "knowledge_retrieval"
            )
            if retrieval_value in {"required", "optional"}:
                normalized["retrieval_requirement"] = retrieval_value
                if (
                    retrieval_value == "required"
                    and "knowledge_retrieval"
                    not in normalized.get("required_capabilities", [])
                ):
                    normalized["required_capabilities"] = [
                        *normalized.get("required_capabilities", []),
                        "knowledge_retrieval",
                    ]
            elif retrieval_value == "forbidden":
                normalized["retrieval_requirement"] = "not_needed"
            citation_value = capability_constraints.get(
                "citation_validation"
            )
            if citation_value in {"required", "preferred"}:
                normalized["citation_requirement"] = citation_value
            for capability in (
                "financial_calculation",
                "complex_reasoning",
                "citation_validation",
            ):
                if (
                    capability_constraints.get(capability) == "required"
                    and capability
                    not in normalized.get("required_capabilities", [])
                ):
                    normalized["required_capabilities"] = [
                        *normalized.get("required_capabilities", []),
                        capability,
                    ]
        else:
            # Legacy payload: synthesize typed constraints from legacy fields.
            normalization_repairs.append(
                "capability_constraints_derived_from_legacy"
            )
            legacy_capability: dict[str, str] = {}
            rr = normalized.get("retrieval_requirement")
            if rr in {"required", "optional", "not_needed"}:
                legacy_capability["knowledge_retrieval"] = rr
            cr = normalized.get("citation_requirement")
            if cr in {"required", "preferred"}:
                legacy_capability["citation_validation"] = (
                    "required" if cr == "required" else "optional"
                )
            normalized["capability_constraints"] = legacy_capability
        normalized["normalization_repairs"] = list(
            dict.fromkeys(normalization_repairs)
        )

        if (
            "memory_read" in normalized.get("required_capabilities", [])
            and capability_constraints.get("memory_read") != "required"
        ):
            # memory_read is never an implicit completion gate: it is only
            # used when the user explicitly requires it.
            normalized["required_capabilities"] = [
                capability
                for capability in normalized["required_capabilities"]
                if capability != "memory_read"
            ]

        for capability, value in normalized["capability_constraints"].items():
            if (
                value == "forbidden"
                and capability in normalized.get("required_capabilities", [])
            ):
                if capability == "memory_read":
                    normalized["required_capabilities"] = [
                        item
                        for item in normalized["required_capabilities"]
                        if item != "memory_read"
                    ]
                    continue
                raise ValueError(
                    f"capability {capability} is both forbidden and required"
                )

        required_capabilities = list(normalized.get("required_capabilities") or [])
        required_financial_tasks = [
            task
            for task in tasks
            if task.get("required", True)
            and "financial_calculation" in (task.get("capabilities") or [])
        ]
        # A required financial task bound to a deterministic evidence tool is
        # an exact-calculation request even when the router model emits a
        # contradictory false flag.  Normalizing this invariant also ensures
        # require_tool policy and fast-path guards see the same semantics.
        if any(task.get("evidence_tool_names") for task in required_financial_tasks):
            normalized["needs_exact_calculation"] = True
        task_capabilities = {
            capability
            for task in tasks
            if task.get("required", True)
            for capability in (task.get("capabilities") or [])
        }
        missing_capabilities = [
            capability
            for capability in required_capabilities
            if capability not in task_capabilities
        ]
        for capability in missing_capabilities:
            if capability == "citation_validation" and retrieval_tasks:
                for task in retrieval_tasks:
                    task["capabilities"] = list(
                        dict.fromkeys(
                            [*(task.get("capabilities") or []), capability]
                        )
                    )
                continue
            tasks.append(
                {
                    "id": f"required_{capability}",
                    "description": f"Complete required capability: {capability}",
                    "required": True,
                    "capabilities": [capability],
                    "evidence_tool_names": [],
                    "requires_citations": capability == "citation_validation",
                    "task_kind": (
                        "validation" if capability == "citation_validation"
                        else "calculation" if capability == "financial_calculation"
                        else "retrieval" if capability == "knowledge_retrieval"
                        else "synthesis" if capability == "complex_reasoning"
                        else "reasoning"
                    ),
                    "depends_on": [],
                }
            )
        return normalized

    @staticmethod
    def _normalize_document_references(
        items: Any,
    ) -> list[dict[str, str]]:
        """Normalize model resource references to typed document references."""

        if not isinstance(items, list):
            return []
        normalized: list[dict[str, str]] = []
        valid_types = {"title", "alias", "filename", "document_id"}
        strength_map = {
            "explicit_required": "required",
            "required": "required",
            "preferred": "preferred",
            "optional": "preferred",
            "mention_only": "mention_only",
        }
        for item in items:
            if isinstance(item, str):
                reference = item.strip()
                reference_type = "title"
                strength = "required"
            elif isinstance(item, dict):
                reference = str(item.get("reference") or "").strip()
                raw_type = str(item.get("reference_type") or "title").strip()
                reference_type = (
                    raw_type if raw_type in valid_types else "title"
                )
                raw_strength = str(
                    item.get("strength") or "required"
                ).strip().lower()
                strength = strength_map.get(raw_strength, "required")
            else:
                continue
            if not reference:
                continue
            normalized.append(
                {
                    "reference": reference[:200],
                    "reference_type": reference_type,
                    "strength": strength,
                }
            )
        return normalized

    @staticmethod
    def _normalize_task_ids(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stable Python-side repair for protocol-only task ids.

        ``id`` is a protocol field, not a semantic field.  Missing ids,
        non-string ids (e.g. ``1``) and duplicates are repaired locally with
        deterministic ``T1..Tn`` ids instead of asking the model to retry.
        Dependencies are remapped through the same old->new id mapping so a
        repaired route never references a vanished id.
        """

        if not tasks:
            return tasks

        seen: set[str] = set()
        needs_renumber = False
        for task in tasks:
            raw = task.get("id")
            if (
                not isinstance(raw, str)
                or not _VALID_TASK_ID_RE.match(raw)
                or raw in seen
            ):
                needs_renumber = True
                break
            seen.add(raw)

        if not needs_renumber:
            return tasks

        mapping: dict[str, str] = {}
        normalized: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            new_id = f"T{index}"
            raw = task.get("id")
            old_key = str(raw) if raw is not None else None
            if old_key is not None and old_key not in mapping:
                mapping[old_key] = new_id
            task = dict(task)
            task["id"] = new_id
            depends_on = task.get("depends_on")
            if isinstance(depends_on, list):
                task["depends_on"] = [
                    mapping.get(str(item), str(item))
                    for item in depends_on
                ]
            normalized.append(task)
        return normalized


_EXECUTABLE_CAPABILITIES = {
    "knowledge_retrieval",
    "financial_calculation",
    "resource_catalog_read",
    "web_search",
}


def validate_semantic_consistency(
    route: SemanticRouteDecision,
) -> list[str]:
    """Structural consistency invariants of the semantic proposal.

    These are protocol-level checks, not natural-language interpretation:
    - state_update_only must not carry executable required capabilities;
    - required retrieval must not be marked not_needed at capability level;
    - clarify mode and needs_clarification must agree.
    """

    violations: list[str] = []
    has_semantic_mutation = bool(
        route.fact_updates
        or route.extracted_facts
        or route.constraint_updates
    )
    has_task_goal = bool((route.resolved_goal or "").strip())
    if (
        not route.state_update_only
        and has_task_goal
        and not route.required_capabilities
        and not route.task_requirements
        and not has_semantic_mutation
    ):
        violations.append(
            "invalid_task_proposal_without_requirements"
        )
    if route.state_update_only and not has_semantic_mutation:
        violations.append("state_update_only_without_mutation")
    task_capabilities = {
        capability
        for task in route.task_requirements
        if task.required
        for capability in task.capabilities
    }
    executable = (
        set(route.required_capabilities) | task_capabilities
    ) & _EXECUTABLE_CAPABILITIES
    if route.state_update_only and executable:
        violations.append(
            "state_update_only_with_executable_capability"
        )
    if route.state_update_only and route.needs_exact_calculation:
        violations.append(
            "state_update_only_with_exact_calculation"
        )
    if (
        route.retrieval_requirement == "required"
        and route.capability_constraints.get(
            "knowledge_retrieval"
        )
        == "not_needed"
    ):
        violations.append(
            "required_retrieval_with_not_needed_capability"
        )
    if route.needs_clarification != (
        route.orchestration_mode == "clarify"
    ):
        violations.append("clarify_mode_inconsistent")
    return violations


def assess_task_admission(
    route: SemanticRouteDecision,
) -> dict[str, Any]:
    """Python-side Task Admission Gate.

    Computes four structural facts from the already-validated typed proposal
    and decides whether this turn admits a Task at all.  No natural-language
    guessing: everything comes from validated fields only.

    kind:
      conversational  -> no Task, no business state mutation
      state_mutation  -> pure semantic state change
      existing_task   -> continuation/reference to an existing Task
      new_task        -> genuinely new Task
    """

    has_semantic_mutation = bool(
        route.fact_updates
        or route.extracted_facts
        or route.constraint_updates
    )
    has_task_goal = bool((route.resolved_goal or "").strip())
    has_execution_requirement = bool(
        route.required_capabilities or route.task_requirements
    )
    task_ref = getattr(route, "task_reference", None)
    task_ref_status = ""
    task_ref_handle = ""
    if task_ref is not None:
        task_ref_data = (
            task_ref.model_dump()
            if hasattr(task_ref, "model_dump")
            else dict(task_ref)
        )
        task_ref_status = str(
            task_ref_data.get("status") or ""
        )
        task_ref_handle = str(
            task_ref_data.get("task_handle") or ""
        )
    has_resolved_task_reference = bool(
        task_ref_status == "resolved" or task_ref_handle
    )
    admitted = bool(
        has_semantic_mutation
        or has_task_goal
        or has_execution_requirement
        or has_resolved_task_reference
    )
    if not admitted:
        kind = "conversational"
    elif has_semantic_mutation:
        kind = "state_mutation"
    elif has_resolved_task_reference:
        kind = "existing_task"
    else:
        kind = "new_task"
    return {
        "admitted": admitted,
        "kind": kind,
        "has_semantic_mutation": has_semantic_mutation,
        "has_task_goal": has_task_goal,
        "has_execution_requirement": has_execution_requirement,
        "has_resolved_task_reference": has_resolved_task_reference,
    }


def conservative_route_fallback(
    *,
    enable_rag: bool,
    allowed_tool_groups: list[str],
    error_type: str,
    requirement_contract: RequestRequirementContract | None = None,
) -> SemanticRouteDecision:
    """Operational fallback that does not reinterpret the user's language."""

    if requirement_contract is not None:
        capabilities = [
            capability
            for capability in requirement_contract.required_capabilities
            if capability != "memory_read"
        ]
        if "complex_reasoning" not in capabilities:
            capabilities.append("complex_reasoning")
        tasks = [
            task
            for task in requirement_contract.task_requirements
            if "memory_read" not in task.capabilities
        ]
        if not any("complex_reasoning" in task.capabilities for task in tasks):
            tasks.append(
                TaskRequirement(
                    id="required_answer_synthesis",
                    description="Synthesize the final answer from completed requirements",
                    capabilities=["complex_reasoning"],
                    evidence_tool_names=[],
                    task_kind="synthesis",
                )
            )
        retrieval = requirement_contract.retrieval_requirement
        citation = requirement_contract.citation_requirement
        calculation = requirement_contract.calculation_requirement == "required"
        return SemanticRouteDecision(
            orchestration_mode=(
                "hybrid" if retrieval != "not_needed" and calculation
                else "rag" if retrieval != "not_needed"
                else "tool" if calculation else "direct"
            ),
            required_capabilities=capabilities,
            task_requirements=tasks,
            retrieval_requirement=retrieval,
            citation_requirement=citation,
            grounding_requirement=("authoritative" if retrieval == "required" else "supplemental" if retrieval == "optional" else "none"),
            retrieval_scope=("all_accessible_knowledge_base" if retrieval != "not_needed" else "none"),
            needs_exact_calculation=requirement_contract.needs_exact_calculation,
            memory_constraint="not_needed",
            confidence=0,
            ambiguities=[f"semantic_router_degraded:{error_type}"],
            reason_summary="Semantic orchestration degraded; immutable user requirements were preserved.",
        )

    capabilities: list[Capability] = ["complex_reasoning"]
    tasks = [
        TaskRequirement(
            id="answer_request",
            description="Answer as safely as possible while semantic routing is degraded",
            capabilities=["complex_reasoning"],
            evidence_tool_names=[],
            requires_citations=False,
            task_kind="synthesis",
        )
    ]
    if "financial_calculation" in set(allowed_tool_groups):
        capabilities.append("financial_calculation")
        tasks.append(
            TaskRequirement(
                id="verified_calculation",
                description="Use authorized deterministic financial calculations when applicable",
                capabilities=["financial_calculation"],
                required=False,
                evidence_tool_names=[],
                requires_citations=False,
                task_kind="calculation",
            )
        )
    if enable_rag:
        capabilities.append("knowledge_retrieval")
        tasks.append(
            TaskRequirement(
                id="knowledge_lookup",
                description="Attempt retrieval from the accessible knowledge base",
                capabilities=["knowledge_retrieval"],
                required=False,
                evidence_tool_names=[],
                requires_citations=False,
                task_kind="retrieval",
            )
        )
    return SemanticRouteDecision(
        # A degraded semantic router must not promote optional caller-enabled
        # capabilities into mandatory user intent. Retrieval remains a safe,
        # optional enrichment attempt; the planner can still use allowed tools.
        orchestration_mode="rag" if enable_rag else "direct",
        required_capabilities=["complex_reasoning"],
        task_requirements=tasks,
        retrieval_requirement="optional" if enable_rag else "not_needed",
        citation_requirement="not_needed",
        grounding_requirement="supplemental" if enable_rag else "none",
        retrieval_scope="all_accessible_knowledge_base" if enable_rag else "none",
        memory_constraint="not_needed",
        confidence=0,
        ambiguities=[f"semantic_router_degraded:{error_type}"],
        reason_summary="Semantic routing failed; using only caller-authorized capabilities.",
    )
