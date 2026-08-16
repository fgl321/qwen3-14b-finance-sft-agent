from __future__ import annotations

from dataclasses import dataclass

from app.control_plane.contracts import merge_contract_draft, repair_draft_from_floor, seal_contract
from app.control_plane.enums import InvocationStatus, PermissionLevel, RequirementLevel
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.metrics import ControlPlaneMetrics
from app.control_plane.reconciler import reconcile_strategy
from app.control_plane.schemas import (
    PreliminaryStrategy,
    RuntimeCapabilitySnapshot,
    ResolvedResourceScope,
    SemanticRequirementContract,
    ShadowControlPlaneResult,
    ToolManifest,
)


@dataclass(frozen=True, slots=True)
class ShadowDiff:
    request_id: str
    production_run_id: str
    production_revision: str
    shadow_revision: str
    production_requirement_summary: dict[str, str]
    shadow_floor_hash: str
    shadow_contract_hash: str | None
    shadow_strategy_hash: str | None
    required_dropped: tuple[str, ...] = ()
    forbidden_planned: tuple[str, ...] = ()
    scope_expansions: tuple[str, ...] = ()
    missing_strategy_capabilities: tuple[str, ...] = ()
    extra_optional_capabilities: tuple[str, ...] = ()
    task_status_disagreement: bool = False
    health_status_disagreement: bool = False
    delivery_status_disagreement: bool = False
    reason_codes: tuple[str, ...] = ()
    side_effect_count: int = 0
    memory_write_count: int = 0

    def __post_init__(self) -> None:
        if self.side_effect_count != 0 or self.memory_write_count != 0:
            raise ValueError("shadow runtime cannot report side effects or memory writes")


class ShadowCapabilityRegistry:
    """Structurally read-only registry: it exposes metadata, never executable callables."""

    def __init__(self, manifests: tuple[ToolManifest, ...] = ()) -> None:
        self._manifests = manifests

    @property
    def manifests(self) -> tuple[ToolManifest, ...]:
        return self._manifests


class ShadowControlPlane:
    revision = "control-plane-shadow-v1"

    def __init__(self, *, registry: ShadowCapabilityRegistry | None = None) -> None:
        self._registry = registry or ShadowCapabilityRegistry()
        self._floor_parser = ExplicitConstraintParser()

    def evaluate(
        self,
        *,
        request_id: str,
        production_run_id: str,
        production_revision: str,
        user_message: str,
        extractor_contract: SemanticRequirementContract | None,
        preliminary_strategy: PreliminaryStrategy | None,
        runtime_snapshot: RuntimeCapabilitySnapshot,
        metrics: ControlPlaneMetrics,
        resolved_scopes: tuple[ResolvedResourceScope, ...] = (),
    ) -> tuple[ShadowControlPlaneResult, ShadowDiff]:
        metrics.observe_request()
        floor = self._floor_parser.parse(request_id=request_id, user_message=user_message)
        draft = merge_contract_draft(floor=floor, extractor=extractor_contract)
        repaired, revision = repair_draft_from_floor(floor=floor, draft=draft)
        if repaired.conflicts:
            conflict_codes = tuple(dict.fromkeys(item.reason_code for item in repaired.conflicts))
            result = ShadowControlPlaneResult(
                request_id=request_id,
                production_run_id=production_run_id,
                shadow_revision=self.revision,
                shadow_floor_hash=floor.canonical_hash,
                shadow_extractor_hash=(extractor_contract.canonical_hash if extractor_contract else None),
                diff_reason_codes=conflict_codes,
            )
            diff = ShadowDiff(
                request_id=request_id,
                production_run_id=production_run_id,
                production_revision=production_revision,
                shadow_revision=self.revision,
                production_requirement_summary={},
                shadow_floor_hash=floor.canonical_hash,
                shadow_contract_hash=None,
                shadow_strategy_hash=None,
                reason_codes=tuple(code.value for code in conflict_codes),
            )
            return result, diff
        contract = seal_contract(repaired)
        strategy = reconcile_strategy(
            run_id=f"shadow:{production_run_id}",
            contract=contract,
            preliminary=preliminary_strategy,
            scopes=resolved_scopes,
            runtime_snapshot=runtime_snapshot,
            tool_manifests=self._registry.manifests,
        )
        required = {
            item.capability for item in contract.constraints
            if item.requirement == RequirementLevel.REQUIRED
        }
        forbidden = {
            item.capability for item in contract.constraints
            if item.permission == PermissionLevel.FORBIDDEN
        }
        planned = {step.capability for step in strategy.steps}
        floor_required = {
            item.capability for item in floor.constraints
            if item.requirement == RequirementLevel.REQUIRED
        }
        required_dropped = tuple(sorted(floor_required - required))
        missing_strategy = tuple(sorted(required - planned))
        forbidden_planned = tuple(sorted(forbidden & planned))
        if required_dropped:
            metrics.increment("required_drop_count", len(required_dropped))
        if forbidden_planned:
            metrics.increment("forbidden_execution_count", len(forbidden_planned))
        if revision is not None:
            metrics.increment("contract_integrity_repair_rate")
        if extractor_contract is not None and extractor_contract.invocation_status in {
            InvocationStatus.PROTOCOL_FAILED, InvocationStatus.SERVICE_FAILED, InvocationStatus.TIMEOUT
        }:
            metrics.increment("extractor_protocol_degraded_rate")
        result = ShadowControlPlaneResult(
            request_id=request_id,
            production_run_id=production_run_id,
            shadow_revision=self.revision,
            shadow_floor_hash=floor.canonical_hash,
            shadow_extractor_hash=(extractor_contract.canonical_hash if extractor_contract else None),
            shadow_contract_hash=contract.canonical_hash,
            shadow_strategy_hash=strategy.canonical_hash,
            sealed_contract=contract.model_dump(mode="json"),
            effective_strategy=strategy.model_dump(mode="json"),
            diff_reason_codes=strategy.reason_codes,
        )
        diff = ShadowDiff(
            request_id=request_id,
            production_run_id=production_run_id,
            production_revision=production_revision,
            shadow_revision=self.revision,
            production_requirement_summary={},
            shadow_floor_hash=floor.canonical_hash,
            shadow_contract_hash=contract.canonical_hash,
            shadow_strategy_hash=strategy.canonical_hash,
            required_dropped=required_dropped,
            forbidden_planned=forbidden_planned,
            missing_strategy_capabilities=missing_strategy,
            reason_codes=tuple(code.value for code in strategy.reason_codes),
        )
        return result, diff
