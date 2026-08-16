from __future__ import annotations

from pydantic import ConfigDict

from app.control_plane.enums import (
    CostEffect,
    DataEffect,
    MutationEffect,
    NetworkEffect,
    SensitiveDataEffect,
)
from app.control_plane.schemas import FrozenModel, ToolEffects


_RANKS = {
    "network": {NetworkEffect.NONE: 0, NetworkEffect.INTERNAL: 1, NetworkEffect.EXTERNAL: 2, NetworkEffect.UNKNOWN: 99},
    "data": {DataEffect.NONE: 0, DataEffect.LOCAL_READ: 1, DataEffect.EXTERNAL_READ: 2, DataEffect.EXTERNAL_WRITE: 3, DataEffect.UNKNOWN: 99},
    "mutation": {MutationEffect.NONE: 0, MutationEffect.REVERSIBLE: 1, MutationEffect.IRREVERSIBLE: 2, MutationEffect.UNKNOWN: 99},
    "sensitive_data": {SensitiveDataEffect.NONE: 0, SensitiveDataEffect.PII_READ: 1, SensitiveDataEffect.PII_EGRESS: 2, SensitiveDataEffect.UNKNOWN: 99},
    "cost": {CostEffect.FREE: 0, CostEffect.METERED: 1, CostEffect.UNKNOWN: 99},
}


class EffectPolicy(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_allowed: ToolEffects = ToolEffects()


def effect_closure(*effects: ToolEffects) -> ToolEffects:
    if not effects:
        return ToolEffects()
    values: dict[str, object] = {}
    for field_name, ranks in _RANKS.items():
        values[field_name] = max(
            (getattr(item, field_name) for item in effects),
            key=lambda value: ranks[value],
        )
    return ToolEffects(**values)


def resolve_unknown_effects(
    resolved: ToolEffects,
    declared_maximum: ToolEffects,
) -> ToolEffects:
    values: dict[str, object] = {}
    for field_name in _RANKS:
        value = getattr(resolved, field_name)
        values[field_name] = (
            getattr(declared_maximum, field_name)
            if str(value.value) == "unknown"
            else value
        )
    return ToolEffects(**values)


def effects_allowed(effects: ToolEffects, policy: EffectPolicy) -> bool:
    for field_name, ranks in _RANKS.items():
        actual = getattr(effects, field_name)
        maximum = getattr(policy.maximum_allowed, field_name)
        if ranks[actual] > ranks[maximum]:
            return False
    return True
