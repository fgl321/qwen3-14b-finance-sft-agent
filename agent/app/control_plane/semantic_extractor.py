from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from app.control_plane.enums import InvocationStatus
from app.control_plane.schemas import SemanticRequirementContract


class IndependentSemanticRequirementExtractor:
    """Structured semantic extractor boundary, independent from routing.

    The injected gateway is responsible only for producing structured data.
    Shadow runtime never supplies it with a tool registry or write capability.
    """

    def __init__(
        self,
        gateway: Callable[[str, str], Awaitable[dict[str, Any]]],
    ) -> None:
        self._gateway = gateway

    async def extract(self, *, request_id: str, user_message: str) -> SemanticRequirementContract:
        try:
            payload = await self._gateway(request_id, user_message)
            contract = SemanticRequirementContract.model_validate(
                {**payload, "request_id": request_id, "invocation_status": InvocationStatus.SUCCESS}
            )
        except (ValidationError, TypeError, ValueError):
            contract = SemanticRequirementContract(
                request_id=request_id,
                invocation_status=InvocationStatus.PROTOCOL_FAILED,
            )
        except (TimeoutError, OSError):
            contract = SemanticRequirementContract(
                request_id=request_id,
                invocation_status=InvocationStatus.SERVICE_FAILED,
            )
        return contract.model_copy(update={"canonical_hash": contract.calculate_hash()})
