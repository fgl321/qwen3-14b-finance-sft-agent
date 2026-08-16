"""Frozen control-plane v2 contracts.

The package is intentionally independent from the production graph until the
shadow feature is enabled.  Business nodes must consume sealed objects rather
than mutate these contracts in place.
"""

from app.control_plane.enums import *  # noqa: F401,F403
from app.control_plane.reason_codes import ReasonCode, reason_definition
from app.control_plane.schemas import *  # noqa: F401,F403

__all__ = ["ReasonCode", "reason_definition"]
"""Frozen v2 control-plane primitives.

The package is side-effect free unless an application explicitly supplies an
executor.  Shadow components intentionally have no executor interface.
"""

from app.control_plane.metrics import ControlPlaneMetrics, RED_LINE_METRICS
from app.control_plane.shadow import ShadowCapabilityRegistry, ShadowControlPlane, ShadowDiff

__all__ = [
    "ControlPlaneMetrics",
    "RED_LINE_METRICS",
    "ShadowCapabilityRegistry",
    "ShadowControlPlane",
    "ShadowDiff",
]
