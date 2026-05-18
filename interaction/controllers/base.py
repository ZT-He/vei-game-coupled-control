"""interaction/controllers/base.py

Common controller interfaces for Step 5 (planner swap integration).

All ego controllers should implement `plan(...)` and return `EgoControlOutput`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from interaction.game.state import VehState, VRUState


@dataclass
class EgoControlOutput:
    """Unified output from any ego controller."""

    a_cmd: float
    delta_cmd: float

    feasible: bool = True
    debug: Dict[str, Any] = None

    def __post_init__(self):
        if self.debug is None:
            self.debug = {}


class EgoController:
    """Abstract ego controller interface."""

    name: str = "base"

    def plan(
        self,
        *,
        veh0: VehState,
        vru0: VRUState,
        delta0: float,
        ref_speed: float,
        detected: bool,
        belief: Any,
        scenario: str,
        **kwargs,
    ) -> EgoControlOutput:
        raise NotImplementedError
