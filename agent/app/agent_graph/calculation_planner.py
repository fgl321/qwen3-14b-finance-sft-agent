from __future__ import annotations

from dataclasses import dataclass
from typing import Any


YEARLY_EXPENSE_TO_MONTHLY_TOOL = "yearly_expense_to_monthly"
EMERGENCY_FUND_RANGE_TOOL = "emergency_fund_range"
LIFE_INSURANCE_GAP_TOOL = "life_insurance_gap"


@dataclass(frozen=True)
class CalculationStep:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CalculationPlan:
    supported: bool
    steps: list[CalculationStep]
    missing_fields: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "steps": [
                {
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                }
                for step in self.steps
            ],
            "missing_fields": list(self.missing_fields),
            "reason": self.reason,
        }
