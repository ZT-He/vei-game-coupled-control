"""interaction/oracle.py

Shared interaction oracle used by all ego controllers.

Design goal
-----------
Given a candidate ego plan, predict the follower (VRU) best-response trajectory
under a fixed follower algorithm (enumerative discrete actions) and type-specific
parameters (cost weights).

This module is intentionally *scenario-agnostic*:
- It only uses the abstract game-layer dynamics and costs.
- Scenario-specific belief updates / perception live outside.

The oracle can be used by:
- Rule-based planners (generate a few candidate ego plans; score with oracle)
- MPC planners (optimize over candidate ego plans; score with oracle)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from interaction.game.state import VehState, VRUState, copy_veh, copy_vru
from interaction.game.dynamics import VehicleAction, VRUAction, step_vehicle, step_vru
from interaction.game.payoff import PayoffWeights, stage_costs


@dataclass
class EgoPlan:
    """Common ego plan format for all controllers.

    For 1D controllers, `delta_seq` can be constant (all zeros).
    For convenience, `a_seq` and `delta_seq` can have length 1 and will be
    repeated to match the oracle horizon.
    """

    a_seq: List[VehicleAction]
    delta_seq: List[float]

    def action_at(self, k: int) -> VehicleAction:
        if not self.a_seq:
            raise ValueError("EgoPlan.a_seq cannot be empty")
        return self.a_seq[k] if k < len(self.a_seq) else self.a_seq[-1]

    def delta_at(self, k: int) -> float:
        if not self.delta_seq:
            return 0.0
        return float(self.delta_seq[k] if k < len(self.delta_seq) else self.delta_seq[-1])


@dataclass
class BestResponseRollout:
    """A single best-response rollout for a fixed VRU type."""

    vru_action_seq: List[VRUAction]
    veh_traj: List[VehState]
    vru_traj: List[VRUState]
    Jv_sum: float
    Js_sum: float
    trace: List[Dict[str, Any]]

    @property
    def a0(self) -> VRUAction:
        return self.vru_action_seq[0] if self.vru_action_seq else VRUAction("go")


class InteractionOracle:
    """Follower best-response oracle.

    The follower algorithm is fixed:
      at each step, pick the VRU action that minimizes the follower stage cost
      given the ego plan action at that step.

    Notes:
    - Ego plan is treated as an open-loop sequence of actions.
    - Vehicle cost is accumulated using `weights_vehicle` but *only* for the
      realized follower best-response action sequence.
    """

    def __init__(self, horizon_steps: int, vru_actions: List[VRUAction]):
        self.H = int(horizon_steps)
        self.AS = list(vru_actions)
        if self.H <= 0:
            raise ValueError("horizon_steps must be positive")
        if not self.AS:
            raise ValueError("vru_actions cannot be empty")

    def rollout_best_response(
        self,
        *,
        veh0: VehState,
        vru0: VRUState,
        ego_plan: EgoPlan,
        ref_speed: float,
        weights_vehicle: PayoffWeights,
        weights_vru: PayoffWeights,
    ) -> BestResponseRollout:
        veh = copy_veh(veh0)
        vru = copy_vru(vru0)

        u_last = float(veh.u_last)

        veh_traj: List[VehState] = [copy_veh(veh)]
        vru_traj: List[VRUState] = [copy_vru(vru)]
        vru_action_seq: List[VRUAction] = []

        Jv_sum = 0.0
        Js_sum = 0.0
        trace: List[Dict[str, Any]] = []

        for k in range(self.H):
            a_v = ego_plan.action_at(k)
            delta = ego_plan.delta_at(k)

            # vehicle evolves independent of follower action (given a_v, delta)
            veh1 = step_vehicle(veh, a_v, delta)

            best_s: Optional[VRUAction] = None
            best_Js: Optional[float] = None
            best_Jv: Optional[float] = None
            best_vru1: Optional[VRUState] = None
            best_dbg: Optional[Dict[str, Any]] = None

            for a_s in self.AS:
                vru1 = step_vru(vru, a_s)

                # Single call: vehicle cost (w) and VRU cost (w_vru) computed
                # together, avoiding duplicate collision/TTC geometry evaluation.
                # Jv_k uses weights_vehicle; Js_k uses weights_vru.
                Jv_k, Js_k, dbg = stage_costs(
                    veh=veh1,
                    vru=vru1,
                    a_v=a_v,
                    a_s=a_s,
                    ref_speed=ref_speed,
                    u_last=u_last,
                    w=weights_vehicle,
                    w_vru=weights_vru,
                )

                if (best_s is None) or (Js_k < float(best_Js)):
                    best_s = a_s
                    best_Js = float(Js_k)
                    best_Jv = float(Jv_k)
                    best_vru1 = vru1
                    best_dbg = dbg

            assert best_s is not None and best_vru1 is not None

            # commit the best-response action for this step
            vru_action_seq.append(best_s)
            Jv_sum += float(best_Jv)
            Js_sum += float(best_Js)

            trace.append(
                {
                    "k": k,
                    "a_v": float(a_v.a),
                    "delta": float(delta),
                    "a_s": str(best_s.mode),
                    "Jv_k": float(best_Jv),
                    "Js_k": float(best_Js),
                    **(best_dbg or {}),
                }
            )

            veh = veh1
            vru = best_vru1
            u_last = float(veh.u_last)

            veh_traj.append(copy_veh(veh))
            vru_traj.append(copy_vru(vru))

        return BestResponseRollout(
            vru_action_seq=vru_action_seq,
            veh_traj=veh_traj,
            vru_traj=vru_traj,
            Jv_sum=float(Jv_sum),
            Js_sum=float(Js_sum),
            trace=trace,
        )
