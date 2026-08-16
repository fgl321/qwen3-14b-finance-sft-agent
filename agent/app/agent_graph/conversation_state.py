from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActiveTaskState(BaseModel):
    """Structured state of the task the conversation is currently doing."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=80)
    goal: str = Field(min_length=1, max_length=500)
    status: Literal[
        "active",
        "awaiting_information",
        "awaiting_confirmation",
        "awaiting_execution",
        "completed",
        "cancelled",
    ] = "active"
    required_capabilities: list[str] = Field(
        default_factory=list,
        max_length=12,
    )
    canonical_facts: list["TaskFact"] = Field(
        default_factory=list,
        max_length=40,
    )
    superseded_facts: list["TaskFact"] = Field(
        default_factory=list,
        max_length=40,
    )
    user_constraints: list["TaskConstraint"] = Field(
        default_factory=list,
        max_length=20,
    )
    source_authority: dict[str, str] = Field(
        default_factory=dict,
        max_length=20,
    )
    derived_results: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    unresolved_requirements: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    created_turn: int = Field(default=1, ge=1)
    updated_turn: int = Field(default=1, ge=1)


class TaskFact(BaseModel):
    """A canonical task fact with active/superseded lifecycle."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=80)
    value: Any = None
    status: Literal["active", "superseded"] = "active"
    source: str = Field(default="current_turn", max_length=60)
    scope: Literal["turn", "task", "session", "durable"] = "task"
    updated_turn: int = Field(default=1, ge=1)


class TaskConstraint(BaseModel):
    """A user constraint attached to the active task (policy-shaped)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=200)
    updated_turn: int = Field(default=1, ge=1)


class FactUpdate(BaseModel):
    """Router-declared current-turn fact patch (Python applies it)."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=80)
    operation: Literal["set", "replace", "clear"] = "set"
    value: Any = None
    source: str = Field(default="current_turn", max_length=60)
    scope: Literal["turn", "task", "session", "durable"] | None = None


class ConstraintUpdate(BaseModel):
    """Router-declared current-turn constraint patch."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=200)
    operation: Literal["set", "clear"] = "set"


class PendingAction(BaseModel):
    """An action the assistant/planner proposed and is waiting to confirm."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=80)
    action_type: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    related_task_handle: str | None = Field(
        default=None,
        max_length=80,
    )
    proposed_by: Literal[
        "assistant",
        "planner",
        "system",
    ] = "assistant"
    status: Literal[
        "pending_confirmation",
        "confirmed",
        "cancelled",
        "completed",
    ] = "pending_confirmation"


class FocusedResource(BaseModel):
    """A resource the conversation is currently focused on."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=80)
    resource_type: str = Field(default="document", max_length=40)
    display_name: str = Field(min_length=1, max_length=300)
    introduced_turn: int = Field(default=1, ge=1)
    last_referenced_turn: int = Field(default=1, ge=1)


class ConversationResultRef(BaseModel):
    """A stable handle to a previous verified result."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=80)
    type: str = Field(min_length=1, max_length=60)
    summary: str = Field(default="", max_length=400)
    created_turn: int = Field(default=1, ge=1)
    calculations: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=20,
    )
    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=40,
    )
    conclusions: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=20,
    )
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=40,
    )
    fact_bindings: list[str] = Field(
        default_factory=list, max_length=40
    )
    derivation_bindings: list[str] = Field(
        default_factory=list, max_length=40
    )
    primary_response_focus: dict[str, Any] | None = Field(
        default=None
    )
    sub_artifact_handles: list[str] = Field(
        default_factory=list, max_length=60
    )
    source_authority: dict[str, str] = Field(
        default_factory=dict,
        max_length=20,
    )


class ConversationState(BaseModel):
    """Structured memory of what this conversation is currently doing."""

    model_config = ConfigDict(extra="forbid")

    active_task: ActiveTaskState | None = None
    focused_resources: list[FocusedResource] = Field(
        default_factory=list,
        max_length=20,
    )
    pending_action: PendingAction | None = None
    recent_results: list[ConversationResultRef] = Field(
        default_factory=list,
        max_length=8,
    )
    last_intent: str | None = Field(default=None, max_length=300)
    turn_count: int = Field(default=0, ge=0)
    # Python-owned mapping: semantic handle -> real document_id.
    resource_handle_map: dict[str, str] = Field(
        default_factory=dict,
        max_length=100,
    )
    next_serials: dict[str, int] = Field(
        default_factory=lambda: {
            "task": 1,
            "action": 1,
            "result": 1,
            "conclusion": 1,
            "claim": 1,
            "calc": 1,
            "doc": 1,
        }
    )
    state_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_references(self) -> "ConversationState":
        allowed_handles = set(self.resource_handle_map)
        for resource in self.focused_resources:
            if resource.handle not in allowed_handles:
                raise ValueError(
                    f"focused resource handle {resource.handle} "
                    "has no document mapping"
                )
        if self.pending_action is not None:
            if self.pending_action.related_task_handle:
                if (
                    self.active_task is None
                    or self.active_task.handle
                    != self.pending_action.related_task_handle
                ):
                    raise ValueError(
                        "pending action related_task_handle must match "
                        "the active task handle"
                    )
        return self


class StateMutationIntent(BaseModel):
    """Compiled intent for every state mutation (single entry channel)."""

    model_config = ConfigDict(extra="forbid")

    target_task: str | None = Field(
        default=None, max_length=80
    )
    fact_mutations: list[dict[str, Any]] = Field(
        default_factory=list, max_length=40
    )
    constraint_mutations: list[dict[str, Any]] = Field(
        default_factory=list, max_length=20
    )
    action_mutations: list[dict[str, Any]] = Field(
        default_factory=list, max_length=20
    )
    expected_state_version: int = Field(default=0, ge=0)


class StateMutationReceipt(BaseModel):
    """Mutation truth: exactly what changed and its before/after values."""

    model_config = ConfigDict(extra="forbid")

    task_handle: str | None = Field(
        default=None, max_length=80
    )
    applied_fact_updates: list[dict[str, Any]] = Field(
        default_factory=list, max_length=40
    )
    applied_constraint_updates: list[dict[str, Any]] = Field(
        default_factory=list, max_length=20
    )
    before_version: int = Field(default=0, ge=0)
    after_version: int = Field(default=0, ge=0)
    status: Literal["committed"] = "committed"


class TurnCommitReceipt(BaseModel):
    """Persistence truth: the whole turn committed atomically or not."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default="", max_length=200)
    thread_id: str = Field(default="", max_length=200)
    before_version: int = Field(default=0, ge=0)
    after_version: int = Field(default=0, ge=0)
    state_hash: str = Field(default="", max_length=64)
    committed: bool = True


class AuthorizedResourceRef(BaseModel):
    """Semantic handle exposed to the model; real identity stays in Python."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1, max_length=80)
    resource_type: str = Field(default="document", max_length=40)
    title: str = Field(min_length=1, max_length=300)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    status: str = Field(default="active", max_length=20)


class CapabilityDescriptor(BaseModel):
    """Model-visible capability; Python still enforces availability."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=300)
    side_effect: bool = False
    evidence_type: str | None = Field(default=None, max_length=60)


class EffectiveOrchestrationDecision(BaseModel):
    """Single source of truth after context resolution; never contradictory."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "direct",
        "execute",
        "state_update_only",
        "plan",
        "clarify",
    ]
    needs_clarification: bool
    reason_code: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_consistency(self) -> "EffectiveOrchestrationDecision":
        if self.mode == "clarify" and not self.needs_clarification:
            raise ValueError(
                "mode=clarify requires needs_clarification=true"
            )
        if (
            self.mode != "clarify"
            and self.needs_clarification
        ):
            raise ValueError(
                "needs_clarification=true requires mode=clarify"
            )
        return self


def default_conversation_state() -> ConversationState:
    return ConversationState()


def _next_serial(state: ConversationState, key: str) -> int:
    serial = int(state.next_serials.get(key) or 1)
    state.next_serials[key] = serial + 1
    return serial


_ARTIFACT_TYPE_HANDLE = {
    "conclusion": ("conclusion", "CONCLUSION_"),
    "claim": ("claim", "CLAIM_"),
    "calc": ("calc", "CALC_"),
    "calculation": ("calc", "CALC_"),
    "action": ("action", "ACTION_"),
}


def allocate_artifact_handle(
    state: ConversationState,
    artifact_type: str,
) -> str:
    """Python-owned allocator: models never create real artifact handles."""

    key, prefix = _ARTIFACT_TYPE_HANDLE.get(
        str(artifact_type).strip().lower(),
        (None, None),
    )
    if key is None or prefix is None:
        raise ValueError(
            f"unknown artifact type: {artifact_type}"
        )
    return f"{prefix}{_next_serial(state, key)}"


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


_CANONICAL_OPERATION_ALIASES = {
    "subtract": "SUBTRACT",
    "minus": "SUBTRACT",
    "sub": "SUBTRACT",
    "-": "SUBTRACT",
    "add": "ADD",
    "plus": "ADD",
    "+": "ADD",
    "multiply": "MULTIPLY",
    "mul": "MULTIPLY",
    "*": "MULTIPLY",
    "x": "MULTIPLY",
    "divide": "DIVIDE",
    "div": "DIVIDE",
    "/": "DIVIDE",
}


def resolve_canonical_operation(raw: Any) -> str | None:
    """Resolve the model's calc operation to the frozen canonical enum.

    Canonical enum (ADD/SUBTRACT/MULTIPLY/DIVIDE/PERCENT/MIN/MAX) is the only
    contract; legacy aliases are accepted only as a defensive compatibility
    layer.  Unsupported spellings return None and the CALC stays unverified.
    """

    op = str(raw or "").strip()
    upper = op.upper()
    if upper in {
        "ADD",
        "SUBTRACT",
        "MULTIPLY",
        "DIVIDE",
        "PERCENT",
        "MIN",
        "MAX",
    }:
        return upper
    return _CANONICAL_OPERATION_ALIASES.get(op.lower())


def _compute_declared_calc(
    artifact: dict[str, Any],
) -> tuple[Any, str]:
    operation = resolve_canonical_operation(
        artifact.get("operation")
    )
    if operation is None:
        return None, "unsupported_operation"
    inputs = artifact.get("inputs")
    values: list[float] = []
    if isinstance(inputs, dict):
        for value in inputs.values():
            number = _coerce_number(value)
            if number is not None:
                values.append(number)
    elif isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict):
                number = _coerce_number(item.get("value"))
                if number is not None:
                    values.append(number)
            else:
                number = _coerce_number(item)
                if number is not None:
                    values.append(number)
    if not values:
        return None, "missing_inputs"
    if operation == "SUBTRACT":
        if len(values) != 2:
            return None, "arity_error"
        return values[0] - values[1], "verified"
    if operation == "ADD":
        return sum(values), "verified"
    if operation == "MULTIPLY":
        result = 1.0
        for value in values:
            result *= value
        return result, "verified"
    if operation == "DIVIDE":
        if len(values) != 2 or values[1] == 0:
            return None, "divide_by_zero"
        return values[0] / values[1], "verified"
    if operation == "PERCENT":
        if len(values) != 2:
            return None, "arity_error"
        return values[0] * values[1] / 100.0, "verified"
    if operation == "MIN":
        return min(values), "verified"
    if operation == "MAX":
        return max(values), "verified"
    return None, "unsupported_operation"


def materialize_new_artifacts(
    state: ConversationState,
    synthesis: dict[str, Any],
) -> dict[str, Any]:
    """Materialize model-proposed artifacts with Python-owned handles.

    The model supplies local_key/artifact_type/text/grounding; the real handle
    (CONCLUSION_n / CLAIM_n / CALC_n) is allocated here and committed only
    with the rest of the turn.
    """

    artifacts: list[dict[str, Any]] = []
    by_local_key: dict[str, dict[str, Any]] = {}
    for raw in (synthesis.get("new_artifacts") or []):
        if not isinstance(raw, dict):
            continue
        local_key = str(
            raw.get("local_key") or ""
        ).strip()
        artifact_type = str(
            raw.get("artifact_type") or "conclusion"
        ).strip().lower()
        if not local_key:
            continue
        handle = allocate_artifact_handle(
            state,
            artifact_type,
        )
        item: dict[str, Any] = {
            "local_key": local_key,
            "handle": handle,
            "artifact_type": artifact_type,
            "text": str(raw.get("text") or "").strip(),
            "grounding": raw.get("grounding"),
            "verification_status": "pending",
        }
        if artifact_type in {"calc", "calculation"}:
            item["operation"] = str(
                raw.get("operation") or ""
            )
            item["inputs"] = raw.get("inputs")
            output, status = _compute_declared_calc(raw)
            item["output"] = output
            item["verification_status"] = status
        artifacts.append(item)
        by_local_key[local_key] = item

    focus: dict[str, Any] | None = None
    candidate = synthesis.get("focus_candidate")
    if isinstance(candidate, dict):
        local_key = str(
            candidate.get("artifact_local_key")
            or candidate.get("local_key")
            or ""
        ).strip()
        item = by_local_key.get(local_key)
        if item is not None:
            focus = {
                "type": item["artifact_type"],
                "handle": item["handle"],
                "local_key": local_key,
            }

    return {
        "artifacts": artifacts,
        "by_local_key": by_local_key,
        "focus": focus,
    }


def validate_referential_integrity(
    *,
    used_fact_refs: list[str],
    used_derivation_ids: list[str],
    used_result_artifact_refs: list[str],
    used_citation_ids: list[str],
    canonical_fact_fields: list[str],
    known_derivation_ids: list[str],
    known_sub_artifact_ids: list[str],
    allowed_citation_ids: list[str],
) -> list[str]:
    """Referential Integrity Gate before commit."""

    violations: list[str] = []
    known_facts = {str(item) for item in canonical_fact_fields}
    known_derivations = {
        str(item) for item in known_derivation_ids
    }
    known_sub_artifacts = {
        str(item) for item in known_sub_artifact_ids
    }
    known_citations = {
        str(item) for item in allowed_citation_ids
    }
    for ref in used_fact_refs:
        if str(ref) not in known_facts:
            violations.append(f"fact:{ref}")
    for ref in used_derivation_ids:
        if str(ref) not in known_derivations:
            violations.append(f"derivation:{ref}")
    for ref in used_result_artifact_refs:
        if str(ref) not in known_sub_artifacts:
            violations.append(f"sub_artifact:{ref}")
    for ref in used_citation_ids:
        if str(ref) not in known_citations:
            violations.append(f"citation:{ref}")
    return violations


def build_resource_catalog(
    candidates: list[dict[str, Any]],
    *,
    state: ConversationState | None = None,
) -> tuple[list[AuthorizedResourceRef], ConversationState]:
    """Build stable semantic handles for the current authorization snapshot.

    Handles from previous turns are reused for the same document_id so
    focused-resource references stay stable across turns.  Python owns the
    handle -> document_id mapping; the model only ever sees handles.
    """

    current_state = (
        state.model_copy(deep=True)
        if state is not None
        else default_conversation_state()
    )
    doc_to_handle = {
        document_id: handle
        for handle, document_id in (
            current_state.resource_handle_map.items()
        )
    }

    catalog: list[AuthorizedResourceRef] = []
    used_handles = set(current_state.resource_handle_map)
    for candidate in sorted(
        candidates,
        key=lambda item: str(item.get("document_id") or ""),
    ):
        document_id = str(candidate.get("document_id") or "")
        if not document_id:
            continue
        handle = doc_to_handle.get(document_id)
        if handle is None:
            handle = f"DOC_{_next_serial(current_state, 'doc')}"
            while handle in used_handles:
                handle = (
                    f"DOC_{_next_serial(current_state, 'doc')}"
                )
            used_handles.add(handle)
        current_state.resource_handle_map[handle] = document_id

        metadata = candidate.get("metadata") or {}
        aliases = (
            list(metadata.get("aliases") or [])
            if isinstance(metadata, dict)
            else []
        )
        catalog.append(
            AuthorizedResourceRef(
                handle=handle,
                resource_type="document",
                title=str(
                    candidate.get("title")
                    or candidate.get("file_name")
                    or document_id
                ),
                aliases=[
                    str(alias)
                    for alias in aliases[:20]
                    if str(alias).strip()
                ],
                status="active",
            )
        )
    return catalog, current_state


def build_capability_catalog() -> list[CapabilityDescriptor]:
    return [
        CapabilityDescriptor(
            name="resource_catalog_read",
            description=(
                "读取当前授权范围内的资源元数据，例如可访问文档数量、"
                "标题、状态和当前资源集合。"
            ),
            side_effect=False,
            evidence_type="system_metadata",
        ),
        CapabilityDescriptor(
            name="knowledge_retrieval",
            description=(
                "从授权文档正文中检索能够支持用户问题的知识和证据。"
            ),
            side_effect=False,
            evidence_type="document_citation",
        ),
        CapabilityDescriptor(
            name="financial_calculation",
            description=(
                "执行确定性数学或金融计算。"
            ),
            side_effect=False,
            evidence_type="tool_result",
        ),
        CapabilityDescriptor(
            name="memory_read",
            description=(
                "读取与当前任务相关的结构化用户记忆。"
            ),
            side_effect=False,
            evidence_type="memory_fact",
        ),
        CapabilityDescriptor(
            name="web_search",
            description=(
                "在用户授权并且任务需要实时外部信息时使用互联网信息。"
            ),
            side_effect=False,
            evidence_type="external_data",
        ),
    ]


def validate_resource_handles(
    *,
    selected_handles: list[str],
    catalog: list[AuthorizedResourceRef],
    allowed_document_ids: list[str],
) -> list[str]:
    """Python-only handle validation: existence, authorization, scope."""

    catalog_by_handle = {
        item.handle: item for item in catalog
    }
    allowed = set(str(item) for item in allowed_document_ids)
    violations: list[str] = []
    for handle in selected_handles:
        entry = catalog_by_handle.get(handle)
        if entry is None:
            violations.append(f"unknown_handle:{handle}")
            continue
        if entry.status != "active":
            violations.append(f"inactive_handle:{handle}")
    return violations


def resource_handles_to_document_ids(
    *,
    selected_handles: list[str],
    catalog: list[AuthorizedResourceRef],
    state: ConversationState,
    allowed_document_ids: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve handles to document ids; unknown/out-of-scope are rejected."""

    violations = validate_resource_handles(
        selected_handles=selected_handles,
        catalog=catalog,
        allowed_document_ids=allowed_document_ids,
    )
    allowed = set(str(item) for item in allowed_document_ids)
    resolved: list[str] = []
    for handle in selected_handles:
        document_id = state.resource_handle_map.get(handle)
        if document_id is None:
            violations.append(f"unknown_handle:{handle}")
            continue
        if document_id not in allowed:
            violations.append(
                f"scope_conflict:{handle}:{document_id}"
            )
            continue
        resolved.append(document_id)
    return list(dict.fromkeys(resolved)), violations


def update_conversation_state(
    *,
    state: ConversationState,
    semantic_route: Any,
    final_answer: str,
    resolved_resources: list[dict[str, Any]] | None = None,
    proposed_action: dict[str, Any] | None = None,
    completed: bool = True,
    admit_task: bool = True,
    result_artifact: dict[str, Any] | None = None,
    response_focus_candidate: dict[str, Any] | None = None,
) -> ConversationState:
    """Deterministic state transition after one completed turn.

    This function only manipulates handles/status; it never reinterprets the
    user's natural language.
    """

    updated = state.model_copy(deep=True)
    updated.turn_count += 1
    turn = updated.turn_count

    pending_resolution = getattr(
        semantic_route, "pending_action_resolution", None
    )
    pending_resolution_data = (
        pending_resolution.model_dump()
        if hasattr(pending_resolution, "model_dump")
        else (
            dict(pending_resolution)
            if pending_resolution
            else {}
        )
    )
    pending_resolution_status = str(
        pending_resolution_data.get("status") or "none"
    )
    if pending_resolution_status == "rejected":
        if updated.pending_action is not None:
            updated.pending_action.status = "cancelled"
        updated.pending_action = None
    elif pending_resolution_status == "confirmed":
        if updated.pending_action is not None:
            updated.pending_action.status = "completed"
            updated.pending_action = None

    if not admit_task:
        # Conversational turn: only bookkeeping (turn_count) changes.
        # No TASK handle, no Result Artifact, no business state mutation.
        updated.last_intent = "conversational"
        return updated

    relation = str(
        getattr(semantic_route, "conversation_relation", None)
        or "new_task"
    )
    resolved_goal = getattr(
        semantic_route, "resolved_goal", None
    )
    task_reference = getattr(
        semantic_route, "task_reference", None
    )
    task_ref_data = (
        task_reference.model_dump()
        if hasattr(task_reference, "model_dump")
        else (dict(task_reference) if task_reference else {})
    )
    task_ref_status = str(
        task_ref_data.get("status") or "none"
    )
    task_ref_handle = str(
        task_ref_data.get("task_handle") or ""
    )

    goal = str(resolved_goal or "").strip()
    if not goal:
        tasks = getattr(semantic_route, "task_requirements", [])
        for task in tasks:
            description = str(getattr(task, "description", "") or "")
            if description:
                goal = description
                break

    capabilities = list(
        getattr(semantic_route, "required_capabilities", [])
    )

    if relation == "cancel_previous":
        if updated.active_task is not None:
            updated.active_task.status = "cancelled"
            updated.active_task.updated_turn = turn
        updated.pending_action = None
        updated.last_intent = "cancel_previous"
        return updated

    if relation in {
        "continuation",
        "follow_up",
        "refinement",
        "correction",
        "confirmation",
        "missing_information_response",
    } and task_ref_status in {"resolved", "ambiguous"}:
        if (
            updated.active_task is not None
            and (
                not task_ref_handle
                or task_ref_handle
                == updated.active_task.handle
            )
        ):
            updated.active_task.updated_turn = turn
            if task_ref_status == "resolved":
                updated.active_task.status = (
                    "completed"
                    if completed
                    else "awaiting_information"
                )
            updated.active_task.required_capabilities = (
                list(
                    dict.fromkeys(
                        [
                            *updated.active_task.required_capabilities,
                            *capabilities,
                        ]
                    )
                )
            )
        elif updated.active_task is None and task_ref_handle:
            updated.active_task = ActiveTaskState(
                handle=task_ref_handle,
                goal=goal or "continue previous task",
                status=(
                    "completed"
                    if completed
                    else "awaiting_information"
                ),
                required_capabilities=capabilities,
                created_turn=turn,
                updated_turn=turn,
            )

    elif (
        updated.active_task is None
        or (
            relation == "new_task"
            and updated.active_task is not None
            and updated.active_task.created_turn != turn
        )
    ):
        if task_ref_handle:
            handle = task_ref_handle
        else:
            handle = f"TASK_{_next_serial(updated, 'task')}"
        updated.active_task = ActiveTaskState(
            handle=handle,
            goal=goal or "unspecified task",
            status=(
                "completed"
                if completed
                else "awaiting_information"
            ),
            required_capabilities=capabilities,
            created_turn=turn,
            updated_turn=turn,
        )
    else:
        if updated.active_task is not None:
            updated.active_task.updated_turn = turn
            updated.active_task.status = (
                "completed"
                if completed
                else "awaiting_information"
            )
            if goal:
                updated.active_task.goal = goal
            updated.active_task.required_capabilities = list(
                dict.fromkeys(
                    [
                        *updated.active_task.required_capabilities,
                        *capabilities,
                    ]
                )
            )

    if updated.active_task is not None:
        route_authority = getattr(
            semantic_route, "source_authority", None
        )
        if route_authority is not None:
            if hasattr(route_authority, "model_dump"):
                updated.active_task.source_authority = (
                    route_authority.model_dump()
                )
            elif isinstance(route_authority, dict):
                updated.active_task.source_authority = dict(
                    route_authority
                )

    if resolved_resources:
        existing = {
            item.handle: item
            for item in updated.focused_resources
        }
        for resource in resolved_resources:
            handle = str(resource.get("handle") or "")
            document_id = str(resource.get("document_id") or "")
            if handle and document_id:
                updated.resource_handle_map[handle] = document_id
            if handle:
                existing.setdefault(
                    handle,
                    FocusedResource(
                        handle=handle,
                        resource_type=str(
                            resource.get("resource_type")
                            or "document"
                        ),
                        display_name=str(
                            resource.get("title")
                            or handle
                        ),
                        introduced_turn=turn,
                        last_referenced_turn=turn,
                    ),
                )
                existing[handle].last_referenced_turn = turn
                existing[handle].display_name = str(
                    resource.get("title")
                    or existing[handle].display_name
                )
        updated.focused_resources = list(
            existing.values()
        )[-20:]

    if proposed_action:
        action_type = str(
            proposed_action.get("action_type") or ""
        ).strip()
        description = str(
            proposed_action.get("description") or ""
        ).strip()
        if action_type and description:
            updated.pending_action = PendingAction(
                handle=(
                    f"ACTION_{_next_serial(updated, 'action')}"
                ),
                action_type=action_type,
                description=description,
                related_task_handle=(
                    updated.active_task.handle
                    if updated.active_task is not None
                    else None
                ),
                proposed_by=str(
                    proposed_action.get("proposed_by")
                    or "assistant"
                ),
                status="pending_confirmation",
            )
            if updated.active_task is not None:
                updated.active_task.status = (
                    "awaiting_confirmation"
                )
                updated.active_task.updated_turn = turn

    if completed and final_answer.strip():
        result_handle = (
            f"RESULT_{_next_serial(updated, 'result')}"
        )
        result_type = _result_type_from_route(semantic_route)
        artifact = dict(result_artifact or {})
        updated.recent_results.insert(
            0,
            ConversationResultRef(
                handle=result_handle,
                type=result_type,
                summary=final_answer.strip()[:400],
                created_turn=turn,
                calculations=list(
                    artifact.get("calculations") or []
                ),
                claims=list(artifact.get("claims") or []),
                conclusions=list(
                    artifact.get("conclusions") or []
                ),
                citations=list(
                    artifact.get("citations") or []
                ),
                fact_bindings=list(
                    artifact.get("fact_bindings") or []
                ),
                derivation_bindings=list(
                    artifact.get("derivation_bindings") or []
                ),
                sub_artifact_handles=list(
                    artifact.get("sub_artifact_handles") or []
                ),
                primary_response_focus=(
                    {
                        "result_handle": result_handle,
                        "type": str(
                            response_focus_candidate.get(
                                "type"
                            )
                            or ""
                        ),
                        "handle": str(
                            response_focus_candidate.get(
                                "handle"
                            )
                            or ""
                        ),
                    }
                    if response_focus_candidate
                    and response_focus_candidate.get("handle")
                    else dict(
                        artifact.get(
                            "primary_response_focus"
                        )
                        or {}
                    )
                    or None
                ),
                source_authority=dict(
                    artifact.get("source_authority") or {}
                ),
            ),
        )
        if updated.active_task is not None:
            updated.active_task.derived_results = list(
                dict.fromkeys(
                    [
                        *updated.active_task.derived_results,
                        *[
                            str(calc.get("handle") or "")
                            for calc in (
                                artifact.get("calculations")
                                or []
                            )
                            if calc.get("handle")
                        ],
                    ]
                )
            )[-20:]
        updated.recent_results = updated.recent_results[:8]

    updated.last_intent = goal or relation
    return updated


def _result_type_from_route(route: Any) -> str:
    capabilities = set(
        getattr(route, "required_capabilities", [])
    )
    if "resource_catalog_read" in capabilities:
        return "resource_catalog"
    if "financial_calculation" in capabilities:
        return "calculation"
    if "knowledge_retrieval" in capabilities:
        return "document_retrieval"
    return "answer"


def build_router_context_block(
    *,
    user_message: str,
    recent_messages: list[dict[str, Any]],
    conversation_state: ConversationState | None,
    resource_catalog: list[AuthorizedResourceRef],
    capability_catalog: list[CapabilityDescriptor],
    scope_snapshot: dict[str, Any] | None,
    narrative_segments: list[dict[str, Any]] | None = None,
) -> str:
    """Structured context block for the context-aware semantic router."""

    state = conversation_state or default_conversation_state()
    context = {
        "current_turn": user_message,
        "recent_messages": [
            {
                "role": str(item.get("role") or "user"),
                "content": str(item.get("content") or "")[:2000],
            }
            for item in recent_messages[-6:]
        ],
        "conversation_state": {
            "active_task": (
                state.active_task.model_dump(mode="json")
                if state.active_task is not None
                else None
            ),
            "pending_action": (
                state.pending_action.model_dump(mode="json")
                if state.pending_action is not None
                else None
            ),
            "focused_resources": [
                item.model_dump(mode="json")
                for item in state.focused_resources
            ],
            "recent_results": [
                item.model_dump(mode="json")
                for item in state.recent_results
            ],
            "turn_count": state.turn_count,
        },
        "authorized_resource_catalog": [
            item.model_dump(mode="json")
            for item in resource_catalog
        ],
        "capability_catalog": [
            item.model_dump(mode="json")
            for item in capability_catalog
        ],
        "current_scope": dict(scope_snapshot or {}),
        "narrative_memory": [
            {
                "segment_id": str(
                    item.get("segment_id") or ""
                ),
                "turn_range": str(
                    item.get("turn_range") or ""
                ),
                "summary": str(item.get("summary") or ""),
            }
            for item in (narrative_segments or [])[-6:]
        ],
    }
    return (
        "<conversation_context>\n"
        f"{json.dumps(context, ensure_ascii=False)}"
        "\n</conversation_context>"
    )


class PolicySnapshot(BaseModel):
    """L0 immutable security/policy snapshot for one request."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(default="default", max_length=200)
    user_id: str = Field(default="anonymous", max_length=200)
    knowledge_base_id: str = Field(
        default="kb_finance_basic", max_length=200
    )
    max_scope_document_ids: list[str] = Field(
        default_factory=list, max_length=200
    )
    allow_web: bool = False
    allow_side_effects: bool = False
    max_agent_rounds: int = Field(default=3, ge=1, le=10)
    max_tool_calls: int = Field(default=12, ge=1, le=50)


class EffectiveTaskContract(BaseModel):
    """Single task-truth source built by Python after the turn patch."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(default="", max_length=500)
    canonical_facts: list[TaskFact] = Field(
        default_factory=list, max_length=40
    )
    superseded_facts: list[TaskFact] = Field(
        default_factory=list, max_length=40
    )
    resolved_resources: list[dict[str, Any]] = Field(
        default_factory=list, max_length=20
    )
    required_capabilities: list[str] = Field(
        default_factory=list, max_length=12
    )
    source_authority: dict[str, str] = Field(
        default_factory=dict, max_length=20
    )
    memory_policy: str = Field(
        default="not_needed",
        pattern="^(not_needed|optional|required|forbidden)$",
    )
    web_policy: str = Field(
        default="forbidden",
        pattern="^(not_needed|optional|required|forbidden)$",
    )
    retrieval_policy: str = Field(
        default="not_needed",
        pattern="^(not_needed|optional|required|forbidden)$",
    )
    user_constraints: list[TaskConstraint] = Field(
        default_factory=list, max_length=20
    )
    result_references: list[dict[str, Any]] = Field(
        default_factory=list, max_length=8
    )
    evidence_requirements: list[str] = Field(
        default_factory=list, max_length=40
    )


FACT_SCOPES = {"turn", "task", "session", "durable"}


def canonicalize_fact_updates(
    updates: list[FactUpdate],
) -> list[FactUpdate]:
    """Collapse same-field mutation duplicates (last write wins).

    The semantic proposal may list the same field in both ``fact_updates``
    and ``extracted_facts`` (for example ``cash set 900000`` twice).  Every
    earlier patch for the same field is superseded by the last one so a
    duplicate write cannot manufacture a bogus superseded history.
    """

    ordered: list[FactUpdate] = []
    for patch in updates:
        field = str(getattr(patch, "field", "") or "").strip()
        if not field:
            continue
        ordered = [
            item
            for item in ordered
            if str(getattr(item, "field", "") or "") != field
        ]
        ordered.append(patch)
    return ordered


def apply_turn_patch(
    *,
    state: ConversationState,
    route: Any,
    turn: int | None = None,
) -> ConversationState:
    """Apply router-declared fact/constraint patches to task state.

    Only handles/state transitions are Python logic; the field names and
    values come from the LLM's structured FactUpdate/ConstraintUpdate.
    """

    updated = state.model_copy(deep=True)
    effective_turn = turn or max(1, updated.turn_count + 1)
    relation = str(
        getattr(route, "conversation_relation", "") or "new_task"
    )
    task_created_this_turn = bool(
        updated.active_task is not None
        and updated.active_task.created_turn == effective_turn
    )
    if updated.active_task is None or (
        relation == "new_task" and not task_created_this_turn
    ):
        previous_task = updated.active_task
        handle = f"TASK_{_next_serial(updated, 'task')}"
        inherited_facts = (
            [
                fact
                for fact in previous_task.canonical_facts
                if fact.scope in {"session", "durable"}
            ]
            if previous_task is not None
            else []
        )
        inherited_superseded = (
            [
                fact
                for fact in previous_task.superseded_facts
                if fact.scope in {"session", "durable"}
            ]
            if previous_task is not None
            else []
        )
        updated.active_task = ActiveTaskState(
            handle=handle,
            goal=str(
                getattr(route, "resolved_goal", "") or ""
            ).strip()
            or "unspecified task",
            status="active",
            required_capabilities=list(
                getattr(route, "required_capabilities", []) or []
            ),
            canonical_facts=inherited_facts,
            superseded_facts=inherited_superseded,
            created_turn=effective_turn,
            updated_turn=effective_turn,
        )
        route_authority = getattr(
            route, "source_authority", None
        )
        if route_authority is not None:
            if hasattr(route_authority, "model_dump"):
                updated.active_task.source_authority = (
                    route_authority.model_dump()
                )
            elif isinstance(route_authority, dict):
                updated.active_task.source_authority = dict(
                    route_authority
                )

    fact_updates = list(
        getattr(route, "fact_updates", []) or []
    )
    if relation == "new_task":
        fact_updates = [
            *fact_updates,
            *list(getattr(route, "extracted_facts", []) or []),
        ]
    fact_updates = canonicalize_fact_updates(fact_updates)
    constraint_updates = list(
        getattr(route, "constraint_updates", []) or []
    )

    active_facts = list(updated.active_task.canonical_facts)
    superseded = list(updated.active_task.superseded_facts)
    for patch in fact_updates:
        field = str(
            getattr(patch, "field", "") or ""
        ).strip()
        if not field:
            continue
        operation = str(
            getattr(patch, "operation", "set") or "set"
        )
        if operation == "clear":
            for fact in list(active_facts):
                if fact.field == field:
                    active_facts.remove(fact)
                    superseded.append(
                        fact.model_copy(
                            update={"status": "superseded"}
                        )
                    )
            continue
        new_value = getattr(patch, "value", None)
        source = str(
            getattr(patch, "source", "") or "current_turn"
        )
        patch_scope = str(getattr(patch, "scope", "") or "")
        explicit_scope = (
            patch_scope if patch_scope in FACT_SCOPES else None
        )
        existing = next(
            (
                fact
                for fact in active_facts
                if fact.field == field
            ),
            None,
        )
        # ResolveFactScope: explicit mutation scope wins; replacing an
        # existing fact without a scope inherits the old scope; first
        # creation without a scope defaults to task.  A value replacement
        # can never silently widen task -> session.
        if explicit_scope is not None:
            effective_scope = explicit_scope
        elif existing is not None:
            effective_scope = existing.scope
        else:
            effective_scope = "task"
        if (
            existing is not None
            and existing.value == new_value
            and existing.scope == effective_scope
        ):
            # Invariant: same-value write is an idempotent no-op and must
            # never manufacture a superseded fact.
            continue
        for fact in list(active_facts):
            if fact.field == field:
                active_facts.remove(fact)
                superseded.append(
                    fact.model_copy(
                        update={"status": "superseded"}
                    )
                )
        active_facts.append(
            TaskFact(
                field=field,
                value=new_value,
                status="active",
                source=source,
                scope=effective_scope,
                updated_turn=effective_turn,
            )
        )
    updated.active_task.canonical_facts = active_facts[:40]
    updated.active_task.superseded_facts = superseded[-40:]

    constraints = list(updated.active_task.user_constraints)
    for patch in constraint_updates:
        name = str(
            getattr(patch, "name", "") or ""
        ).strip()
        if not name:
            continue
        operation = str(
            getattr(patch, "operation", "set") or "set"
        )
        value = str(getattr(patch, "value", "") or "")
        constraints = [
            item
            for item in constraints
            if item.name != name
        ]
        if operation == "set" and value:
            constraints.append(
                TaskConstraint(
                    name=name,
                    value=value,
                    updated_turn=effective_turn,
                )
            )
    updated.active_task.user_constraints = constraints[-20:]
    updated.active_task.updated_turn = effective_turn
    return updated


def build_effective_task_contract(
    *,
    state: ConversationState,
    route: Any,
    resolved_resources: list[dict[str, Any]],
    evidence_requirements: list[str] | None = None,
) -> EffectiveTaskContract:
    """Build the single task-truth source for this turn."""

    def resolve_effective_policy(
        *,
        source_permission: str,
        capability_requirement: str,
    ) -> str:
        """Typed Contract Boundary resolver.

        SourcePermission (allowed/forbidden) answers CAN; CapabilityRequirement
        (not_needed/optional/required/forbidden) answers NEED.  The composed
        EffectivePolicy (not_needed/optional/required/forbidden) answers RUN.
        A forbidden source or forbidden capability always wins.
        """

        if source_permission == "forbidden":
            return "forbidden"
        if capability_requirement == "forbidden":
            return "forbidden"
        if capability_requirement in {
            "not_needed",
            "optional",
            "required",
        }:
            return capability_requirement
        return "forbidden"

    active_task = state.active_task
    canonical_facts: list[TaskFact] = []
    superseded_facts: list[TaskFact] = []
    user_constraints: list[TaskConstraint] = []
    source_authority: dict[str, str] = {}
    goal = ""
    if active_task is not None:
        canonical_facts = list(active_task.canonical_facts)
        superseded_facts = list(active_task.superseded_facts)
        user_constraints = list(active_task.user_constraints)
        source_authority = dict(active_task.source_authority)
        goal = active_task.goal

    route_authority = getattr(route, "source_authority", None)
    constraint_updates = list(
        getattr(route, "constraint_updates", []) or []
    )
    relation = str(
        getattr(route, "conversation_relation", "") or "new_task"
    )
    inherit_authority = bool(
        relation
        in {
            "continuation",
            "follow_up",
            "refinement",
            "correction",
            "confirmation",
            "missing_information_response",
            "cancel_previous",
        }
        and active_task is not None
        and bool(active_task.source_authority)
    )
    if inherit_authority:
        source_authority = dict(active_task.source_authority)
    elif route_authority is not None:
        if hasattr(route_authority, "model_dump"):
            source_authority = route_authority.model_dump()
        elif isinstance(route_authority, dict):
            source_authority = dict(route_authority)
    elif not source_authority:
        source_authority = {
            "current_user_facts": "allowed",
            "selected_documents": "allowed",
            "deterministic_derivation": "allowed",
            "memory": "allowed",
            "general_model_knowledge": "allowed",
            "domain_heuristics": "allowed",
            "web": "forbidden",
        }

    _CONSTRAINT_AUTHORITY_MAP = {
        "web": "web",
        "memory": "memory",
        "general_model_knowledge": "general_model_knowledge",
        "domain_heuristics": "domain_heuristics",
        "selected_documents": "selected_documents",
        "current_user_facts": "current_user_facts",
        "deterministic_derivation": "deterministic_derivation",
    }
    for patch in constraint_updates:
        name = str(
            getattr(patch, "name", "") or ""
        ).strip().lower()
        key = _CONSTRAINT_AUTHORITY_MAP.get(name)
        if key is None:
            continue
        operation = str(
            getattr(patch, "operation", "set") or "set"
        )
        value = str(
            getattr(patch, "value", "") or ""
        ).strip().lower()
        if operation == "clear":
            source_authority[key] = "forbidden"
        elif value in {"allowed", "forbidden"}:
            source_authority[key] = value

    goal = (
        str(getattr(route, "resolved_goal", "") or "").strip()
        or goal
    )
    capabilities = list(
        getattr(route, "required_capabilities", []) or []
    )
    memory_requirement = str(
        getattr(route, "memory_constraint", "not_needed")
        or "not_needed"
    )
    memory_policy = resolve_effective_policy(
        source_permission=source_authority.get(
            "memory", "allowed"
        ),
        capability_requirement=memory_requirement,
    )
    web_requirement = str(
        (
            getattr(route, "capability_constraints", {})
            or {}
        ).get("web_search", "not_needed")
        or "not_needed"
    )
    web_policy = resolve_effective_policy(
        source_permission=source_authority.get(
            "web", "forbidden"
        ),
        capability_requirement=web_requirement,
    )
    retrieval_requirement = str(
        getattr(route, "retrieval_requirement", "not_needed")
        or "not_needed"
    )
    retrieval_policy = resolve_effective_policy(
        source_permission=source_authority.get(
            "selected_documents", "allowed"
        ),
        capability_requirement=retrieval_requirement,
    )
    result_references = [
        item.model_dump()
        if hasattr(item, "model_dump")
        else dict(item)
        for item in (getattr(route, "result_references", []) or [])
    ]

    return EffectiveTaskContract(
        goal=goal,
        canonical_facts=canonical_facts,
        superseded_facts=superseded_facts,
        resolved_resources=resolved_resources,
        required_capabilities=capabilities,
        source_authority=source_authority,
        memory_policy=memory_policy,
        web_policy=web_policy,
        retrieval_policy=retrieval_policy,
        user_constraints=user_constraints,
        result_references=result_references,
        evidence_requirements=list(evidence_requirements or []),
    )


_FACT_PRECEDENCE = {
    "current_turn_correction": 8,
    "current_turn": 7,
    "task_canonical": 6,
    "task_derived": 5,
    "session": 4,
    "long_term_memory": 2,
    "general_model_knowledge": 1,
}


def reconcile_facts(
    *,
    contract: EffectiveTaskContract,
    memory_facts: list[dict[str, Any]] | None = None,
) -> tuple[list[TaskFact], list[str]]:
    """Merge facts with a fixed precedence; LTM never beats task facts.

    Returns (reconciled active facts, shadowed fact descriptions).
    """

    reconciled: list[TaskFact] = []
    shadowed: list[str] = []
    by_field: dict[str, TaskFact] = {}

    for fact in contract.canonical_facts:
        if fact.status != "active":
            continue
        source_priority = (
            "current_turn"
            if fact.source == "current_turn"
            else "task_canonical"
        )
        priority = _FACT_PRECEDENCE.get(source_priority, 6)
        existing = by_field.get(fact.field)
        existing_priority = 0
        if existing is not None:
            existing_source = (
                "current_turn"
                if existing.source == "current_turn"
                else "task_canonical"
            )
            existing_priority = _FACT_PRECEDENCE.get(
                existing_source, 6
            )
        if existing is None or priority >= existing_priority:
            by_field[fact.field] = fact

    if (
        contract.memory_policy != "forbidden"
        and memory_facts
    ):
        for item in memory_facts:
            field = str(item.get("fact_key") or "").strip()
            if not field:
                continue
            value = item.get("fact_value")
            if field in by_field:
                shadowed.append(
                    f"long_term_memory:{field} shadowed by task fact"
                )
                continue
            by_field[field] = TaskFact(
                field=field,
                value=value,
                status="active",
                source="long_term_memory",
                scope="durable",
                updated_turn=1,
            )

    reconciled = list(by_field.values())
    return reconciled, shadowed


class MemoryPromotionGate(BaseModel):
    """Deterministic gate deciding which L2 facts may promote to L1."""

    model_config = ConfigDict(extra="forbid")

    allowed_fact_types: list[str] = Field(
        default_factory=lambda: [
            "identity",
            "household",
            "preference",
            "goal",
            "personal",
        ]
    )
    blocked_key_prefixes: list[str] = Field(
        default_factory=lambda: [
            "calc:",
            "result:",
            "calculation:",
        ]
    )

    def may_promote(
        self,
        *,
        fact_type: str,
        fact_key: str,
        fact_value: Any,
    ) -> tuple[bool, str]:
        clean_type = str(fact_type or "").strip().lower()
        clean_key = str(fact_key or "").strip().lower()
        if clean_type not in self.allowed_fact_types:
            return False, f"fact_type_not_promotable:{clean_type}"
        if any(
            clean_key.startswith(prefix)
            for prefix in self.blocked_key_prefixes
        ):
            return False, f"ephemeral_key:{clean_key}"
        if isinstance(fact_value, dict) and (
            "amount" in fact_value or "result" in fact_value
        ):
            return False, "computed_value_not_promotable"
        return True, "ok"


def build_result_artifact(
    *,
    result: dict[str, Any],
    route: Any,
) -> dict[str, Any]:
    """Structured result artifact from verified execution outputs."""

    calculations: list[dict[str, Any]] = []
    for calc_index, tool_result in enumerate(
        result.get("tool_results") or [],
        start=1,
    ):
        if not bool(tool_result.get("success")):
            continue
        calculations.append(
            {
                "handle": f"CALC_{calc_index}",
                "tool_call_id": tool_result.get("tool_call_id"),
                "tool_name": tool_result.get("tool_name"),
                "output": tool_result.get("output"),
                "source_class": "deterministic_derivation",
            }
        )

    claims: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    claim_index = 1
    final_response_result = (
        result.get("final_response_result") or {}
    )
    if isinstance(final_response_result, str):
        try:
            final_response_result = json.loads(
                final_response_result
            )
        except Exception:
            final_response_result = {}
    if not isinstance(final_response_result, dict):
        final_response_result = {}
    synthesis = (
        final_response_result.get("synthesis") or {}
        if isinstance(final_response_result, dict)
        else {}
    )
    if not isinstance(synthesis, dict):
        synthesis = {}
    fact_bindings = [
        str(item)
        for item in (synthesis.get("used_fact_refs") or [])
    ]
    derivation_bindings = [
        str(item)
        for item in (synthesis.get("used_derivation_ids") or [])
    ]
    used_result_artifact_refs = [
        str(item)
        for item in (
            synthesis.get("used_result_artifact_refs") or []
        )
    ]
    claim_bindings = synthesis.get("claim_bindings") or []
    if not isinstance(claim_bindings, list):
        claim_bindings = []
    primary_response_focus = (
        synthesis.get("primary_response_focus") or {}
    )
    if not isinstance(primary_response_focus, dict):
        primary_response_focus = {}
    used_citations = list(
        synthesis.get("used_citation_ids") or []
    )
    for citation in result.get("citations") or (
        (result.get("rag") or {}).get("citations") or []
    ):
        if not isinstance(citation, dict):
            continue
        citation_id = str(citation.get("citation_id") or "")
        if citation_id and citation_id in {
            str(item) for item in used_citations
        }:
            evidence_text = str(
                citation.get("quote")
                or citation.get("text")
                or (citation.get("metadata") or {}).get(
                    "evidence_excerpt"
                )
                or ""
            ).strip()
            if len(evidence_text) > 500:
                evidence_text = (
                    evidence_text[:500] + "...[truncated]"
                )
            citations.append(
                {
                    "citation_id": citation_id,
                    "document_id": citation.get("document_id"),
                    "file_name": citation.get("file_name"),
                    "text": evidence_text,
                }
            )
            claims.append(
                {
                    "handle": f"CLAIM_{claim_index}",
                    "text": evidence_text,
                    "grounding_type": "document_citation",
                    "citation_ids": [citation_id],
                    "citation_id": citation_id,
                    "document_id": citation.get("document_id"),
                }
            )
            claim_index += 1

    conclusions: list[dict[str, Any]] = []
    case_verdicts = synthesis.get("case_verdicts") or {}
    if isinstance(case_verdicts, dict):
        for index, (case_id, verdict) in enumerate(
            case_verdicts.items(),
            start=1,
        ):
            conclusions.append(
                {
                    "handle": f"CONCLUSION_{index}",
                    "case_id": case_id,
                    "verdict": verdict,
                }
            )

    source_authority: dict[str, str] = {}
    route_authority = getattr(route, "source_authority", None)
    if route_authority is not None:
        if hasattr(route_authority, "model_dump"):
            source_authority = route_authority.model_dump()
        elif isinstance(route_authority, dict):
            source_authority = dict(route_authority)

    sub_artifact_handles = [
        *[str(item.get("handle") or "") for item in calculations],
        *[str(item.get("handle") or "") for item in claims],
        *[str(item.get("handle") or "") for item in conclusions],
    ]

    return {
        "calculations": calculations,
        "claims": claims,
        "conclusions": conclusions,
        "citations": citations,
        "fact_bindings": fact_bindings,
        "derivation_bindings": derivation_bindings,
        "used_result_artifact_refs": used_result_artifact_refs,
        "claim_bindings": claim_bindings,
        "primary_response_focus": primary_response_focus,
        "sub_artifact_handles": sub_artifact_handles,
        "source_authority": source_authority,
    }
