"""interaction/controllers/factory.py

Factory for constructing swappable ego controllers.

Used by:
- qual_sim/sim_vei_intersection.py
- qual_sim/sim_vei_straight_road.py
"""

from __future__ import annotations

from typing import List

from interaction.oracle import InteractionOracle
from interaction.game.dynamics import VehicleAction, VRUAction
from interaction.game.payoff import PayoffWeights

from interaction.controllers.base import EgoController
from interaction.controllers.mpc_1d_stackelberg import MPC1DStackelbergController
from interaction.controllers.rule_based_1d import RuleBased1DController
from interaction.controllers.rule_based_2d import RuleBased2DController
from interaction.controllers.mpc_2d import MPC2DController


def make_ego_controller(
    name: str,
    *,
    oracle: InteractionOracle,
    vehicle_actions: List[VehicleAction],
    weights_vehicle: PayoffWeights,
    verbose: bool = False,
) -> EgoController:
    key = (name or "mpc1d").lower().strip()

    if key in ("rule1d", "1d-rule", "rule-based-1d"):
        return RuleBased1DController(
            oracle=oracle,
            weights_vehicle=weights_vehicle,
        )

    if key in ("rule2d", "2d-rule", "rule-based-2d"):
        return RuleBased2DController(
            oracle=oracle,
            weights_vehicle=weights_vehicle,
        )

    if key in ("mpc1d", "1d-mpc", "game_mpc", "stackelberg_mpc"):
        return MPC1DStackelbergController(
            oracle=oracle,
            vehicle_actions=vehicle_actions,
            weights_vehicle=weights_vehicle,
            verbose=verbose,
        )

    if key in ("mpc2d", "2d-mpc"):
        return MPC2DController(
            oracle=oracle,
            vehicle_actions=vehicle_actions,
            weights_vehicle=weights_vehicle,
            verbose=verbose,
        )

    raise ValueError(
        f"Unknown ego controller '{name}'. "
        "Choose from: mpc1d, rule1d, rule2d, mpc2d."
    )
