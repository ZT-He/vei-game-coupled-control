# interaction/game/stackelberg.py # TODO: check best-response selection + tie-breaking + feasibility flags
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from .state import VehState, VRUState, copy_veh, copy_vru
from .dynamics import VehicleAction, VRUAction, step_vehicle, step_vru
from .payoff import PayoffWeights, stage_costs


@dataclass
class StackelbergResult:
    best_vehicle: VehicleAction
    best_vru: VRUAction
    J_vehicle: float
    J_vru: float
    table: List[Dict[str, Any]]  # debug table rows


class StackelbergOneStep:
    """
    One-step Stackelberg: enumerate A_v; follower chooses best response; leader chooses best.
    """

    def __init__(
        self,
        vehicle_actions: List[VehicleAction],
        vru_actions: List[VRUAction],
        weights_vehicle: PayoffWeights,
        weights_vru: PayoffWeights,
    ):
        self.vehicle_actions = vehicle_actions
        self.vru_actions = vru_actions
        self.wv = weights_vehicle
        self.ws = weights_vru

    def solve(
        self,
        veh0: VehState,
        vru0: VRUState,
        delta0: float,
        ref_speed: float,
    ) -> StackelbergResult:
        rows: List[Dict[str, Any]] = []

        best = None

        for a_v in self.vehicle_actions:
            # follower best response for this a_v
            best_resp = None

            for a_s in self.vru_actions:
                # rollout one step (explicit)
                veh1 = step_vehicle(veh0, a_v, delta0)
                vru1 = step_vru(vru0, a_s)

                Jv, Js, dbg = stage_costs(
                    veh=veh1, vru=vru1,
                    a_v=a_v, a_s=a_s,
                    ref_speed=ref_speed,
                    u_last=veh0.u_last,
                    w=self.wv  # vehicle’s cost weights
                )

                # follower cost uses its own weights: recompute quickly by swapping w
                _, Js2, _ = stage_costs(
                    veh=veh1, vru=vru1,
                    a_v=a_v, a_s=a_s,
                    ref_speed=ref_speed,
                    u_last=veh0.u_last,
                    w=self.ws
                )

                rows.append({
                    "a_v": a_v.a,
                    "a_s": a_s.mode,
                    "J_vehicle": Jv,
                    "J_vru": Js2,
                    **dbg
                })

                if best_resp is None or Js2 < best_resp[0]:
                    best_resp = (Js2, a_s, Jv)

            assert best_resp is not None
            Js_star, a_s_star, Jv_given_resp = best_resp

            if best is None or Jv_given_resp < best[0]:
                best = (Jv_given_resp, a_v, a_s_star, Js_star)

        assert best is not None
        Jv_star, a_v_star, a_s_star, Js_star = best

        return StackelbergResult(
            best_vehicle=a_v_star,
            best_vru=a_s_star,
            J_vehicle=Jv_star,
            J_vru=Js_star,
            table=rows
        )
