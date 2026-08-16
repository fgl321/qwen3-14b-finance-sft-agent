from __future__ import annotations

import re
from dataclasses import dataclass

from app.control_plane.canonical import content_hash
from app.control_plane.enums import Authority, EnforcementStrength, PermissionLevel, RequirementLevel
from app.control_plane.schemas import (
    CapabilityConstraint,
    ConstraintSource,
    ExplicitRequirementFloor,
    ResourceConstraints,
)


@dataclass(frozen=True, slots=True)
class _ExplicitRule:
    rule_id: str
    pattern: re.Pattern[str]
    capability: str
    requirement: RequirementLevel
    permission: PermissionLevel
    scope_ref: str | None = None


# The Floor is a security/protocol boundary, not a natural-language parser.
# Complex NL semantics ("必须检索", "不要使用其他文档", "除A以外只准看A") are
# owned by the Semantic Router; Python only enforces API/server boundaries,
# resource identity, authorization and execution safety.  Keeping regex-based
# NL rules here created two competing semantic authorities that fought each
# other and produced false CONTRACT_PERMISSION_CONFLICTs.
_RULES: tuple[_ExplicitRule, ...] = ()


class ExplicitConstraintParser:
    version = "explicit-floor-v1"

    def parse(self, *, request_id: str, user_message: str) -> ExplicitRequirementFloor:
        constraints: list[CapabilityConstraint] = []
        seen: set[tuple[str, RequirementLevel, PermissionLevel, str | None]] = set()
        for index, rule in enumerate(_RULES, start=1):
            match = rule.pattern.search(user_message)
            if match is None:
                continue
            semantic_key = (
                rule.capability,
                rule.requirement,
                rule.permission,
                rule.scope_ref,
            )
            if semantic_key in seen:
                continue
            seen.add(semantic_key)
            source_text = match.group(0)
            source = ConstraintSource(
                constraint_id=f"floor_{index}_{rule.rule_id}",
                authority=Authority.USER_EXPLICIT,
                enforcement_strength=EnforcementStrength.EXPLICIT_CONSTRAINT,
                rule_id=rule.rule_id,
                source_start=match.start(),
                source_end=match.end(),
                source_hash=content_hash({"text": source_text}),
            )
            constraints.append(
                CapabilityConstraint(
                    capability=rule.capability,
                    requirement=rule.requirement,
                    permission=rule.permission,
                    source=source,
                    scope_ref=rule.scope_ref,
                )
            )
        floor = ExplicitRequirementFloor(
            request_id=request_id,
            constraints=tuple(constraints),
            parser_version=self.version,
        )
        return floor.model_copy(update={"canonical_hash": floor.calculate_hash()})
