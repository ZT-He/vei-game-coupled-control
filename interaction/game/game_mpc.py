# interaction/game/game_mpc.py TODO: Check horizon logic
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .state import VehState, VRUState, copy_veh, copy_vru
from .dynamics import VehicleAction, VRUAction, step_vehicle, step_vru
from .payoff import PayoffWeights, stage_costs


@dataclass
class GameMPCOutput:
    a_vehicle: float
    vru_action_0: VRUAction
    feasible: bool
    debug: Dict[str, Any]


def rider_weights(rider_type: str) -> PayoffWeights:
    """
    Rider style affects VRU objective:
      - Aggressive: prioritize progress; tolerate smaller TTC
      - Conservative: prioritize safety (TTC); willing to yield
    """
    rt = (rider_type or "Conservative").lower()

    if "aggr" in rt:
        return PayoffWeights(
            # vehicle terms unused for VRU, but keep default placeholders
            w_collision_v=1e6, w_ttc_v=2e3, w_speed_v=5.0, w_acc_v=0.5, w_jerk_v=0.2,

            # VRU terms
            w_collision_s=8e5,
            w_ttc_s=1.0e3,        # low risk aversion
            w_progress_s=8.0,     # high progress incentive
            w_effort_s=0.30       # yielding “feels costly”
        )
    else:
        return PayoffWeights(
            w_collision_v=1e6, w_ttc_v=2e3, w_speed_v=5.0, w_acc_v=0.5, w_jerk_v=0.2,

            w_collision_s=2e6,
            w_ttc_s=2.0e4,        # high risk aversion (key difference)
            w_progress_s=3.0,     # lower progress incentive
            w_effort_s=0.05       # yielding is “cheap”
        )

# LEGACY — SimpleGameMPC is superseded by MPC1DStackelbergController
# (interaction/controllers/mpc_1d_stackelberg.py). Not used in the current project.
# Retained below for reference only; class is renamed with a leading underscore
# so it is excluded from public API and clearly marked as inactive.
class _SimpleGameMPC:
    """
    Receding-horizon Stackelberg Game-MPC (enumeration version, easy to debug):
    - Perception gating affects VEHICLE action (leader) via detected flag
    - VRU action still computed (so V2X quality can create good/bad outcomes)
    """

    def __init__(
        self,
        horizon_steps: int,
        vehicle_actions: List[VehicleAction],
        vru_actions: List[VRUAction],
        base_weights_vehicle: PayoffWeights,
        verbose: bool = False
    ):
        self.H = int(horizon_steps)
        self.AV = vehicle_actions # self.AV is a *list* of VehicleAction
        self.AS = vru_actions  # self.AS is a *list* of VRUAction, e.g., {"go","yield"}
        self.base_wv = base_weights_vehicle
        self.verbose = verbose

    def plan(
            self,
            veh0: VehState,
            vru0: VRUState,
            delta0: float,
            ref_speed: float,
            detected: bool,
            belief,  # <-- RiderTypeBelief (or a dict)
            scenario: str = "straight",  # or "intersection"
            vru_react_when_undetected: bool = False,  # NEW
    ) -> GameMPCOutput:

        if belief is None:
            raise ValueError("belief must be provided (RiderTypeBelief or probability dict)")

        wv = self.base_wv

        # belief probabilities
        if hasattr(belief, "probs"):
            b = belief.probs()
        else:
            b = {"Aggressive": float(belief.get("Aggressive", 0.5)),
                 "Conservative": float(belief.get("Conservative", 0.5))}

        type_list = ["Aggressive", "Conservative"]

        if not detected:
            # Keep legacy behavior unless explicitly enabled
            if not vru_react_when_undetected:
                return GameMPCOutput(
                    a_vehicle=0.0,
                    vru_action_0=VRUAction("go"),
                    feasible=True,
                    debug={"detected": False, "policy": "veh_cruise_vru_go"}
                )

            # NEW: AV cruises, but VRU still chooses best-response (by rider type)
            a_v_fixed = VehicleAction(0.0)  # or VehicleAction(a=0.0) depending on your dataclass

            traces_by_type = {}
            a0_by_type = {}
            Js_sum_by_type = {}

            for rt in type_list:
                ws = rider_weights(rt)

                veh = copy_veh(veh0)
                vru = copy_vru(vru0)

                u_last = veh.u_last
                a0_follow = None
                Js_sum = 0.0
                trace = []

                for k in range(self.H):
                    best_s = None
                    best_s_Js = None
                    best_dbg = None

                    for a_s in self.AS:
                        veh1 = step_vehicle(veh, a_v_fixed, delta0)
                        vru1 = step_vru(vru, a_s)

                        # VRU objective only (leader is fixed cruise)
                        _, Js_k, dbg = stage_costs(
                            veh=veh1, vru=vru1, a_v=a_v_fixed, a_s=a_s,
                            ref_speed=ref_speed, u_last=u_last, w=ws
                        )

                        if best_s is None or Js_k < best_s_Js:
                            best_s = a_s
                            best_s_Js = Js_k
                            best_dbg = dbg

                    if k == 0:
                        a0_follow = best_s

                    # apply evolution under chosen follower action
                    veh = step_vehicle(veh, a_v_fixed, delta0)
                    vru = step_vru(vru, best_s)

                    Js_sum += float(best_s_Js)
                    u_last = veh.u_last

                    trace.append({
                        "k": k,
                        "type": rt,
                        "a_v": a_v_fixed.a,
                        "a_s": best_s.mode,
                        "Js_k": float(best_s_Js),
                        **(best_dbg or {})
                    })

                # store results
                traces_by_type[rt] = trace
                Js_sum_by_type[rt] = float(Js_sum)
                a0_by_type[rt] = a0_follow if a0_follow is not None else VRUAction("go")

            # Choose "most-likely" follower action for returned vru_action_0 (legacy compatibility)
            rt_ml = "Aggressive" if b["Aggressive"] >= b["Conservative"] else "Conservative"
            vru_a0_ml = a0_by_type[rt_ml]

            return GameMPCOutput(
                a_vehicle=0.0,
                vru_action_0=vru_a0_ml,
                feasible=True,
                debug={
                    "detected": False,
                    "policy": "veh_cruise_vru_bestresp",
                    "belief": b,
                    "rider_type_ml": rt_ml,
                    # IMPORTANT: keep this key for your quant_game_intersection hook
                    "a0_by_type": {k: v.mode for k, v in a0_by_type.items()},
                    "Js_sum_by_type": Js_sum_by_type,
                    "trace_by_type": traces_by_type,
                }
            )

        best = None
        best_trace = None
        best_a0_by_type = None

        for a_v in self.AV:
            Jv_exp = 0.0
            traces_by_type = {}
            a0_by_type = {}

            for rt in type_list:
                ws = rider_weights(rt)

                veh = copy_veh(veh0)
                vru = copy_vru(vru0)

                Jv_sum = 0.0
                Js_sum = 0.0
                trace = []

                u_last = veh.u_last
                a0_follow = None

                for k in range(self.H):
                    # follower best response under this rider type
                    best_s = None
                    best_s_Js = None
                    best_s_Jv = None
                    best_dbg = None

                    for a_s in self.AS:
                        veh1 = step_vehicle(veh, a_v, delta0)
                        vru1 = step_vru(vru, a_s)

                        Jv_k, _, dbg = stage_costs(
                            veh=veh1, vru=vru1, a_v=a_v, a_s=a_s,
                            ref_speed=ref_speed, u_last=u_last, w=wv
                        )
                        _, Js_k, _ = stage_costs(
                            veh=veh1, vru=vru1, a_v=a_v, a_s=a_s,
                            ref_speed=ref_speed, u_last=u_last, w=ws
                        )

                        if best_s is None or Js_k < best_s_Js:
                            best_s = a_s
                            best_s_Js = Js_k
                            best_s_Jv = Jv_k
                            best_dbg = dbg

                    if k == 0:
                        a0_follow = best_s

                    veh = step_vehicle(veh, a_v, delta0)
                    vru = step_vru(vru, best_s)

                    Jv_sum += float(best_s_Jv)
                    Js_sum += float(best_s_Js)
                    u_last = veh.u_last

                    trace.append({
                        "k": k,
                        "type": rt,
                        "a_v": a_v.a,
                        "a_s": best_s.mode,
                        "Jv_k": float(best_s_Jv),
                        "Js_k": float(best_s_Js),
                        **(best_dbg or {})
                    })

                traces_by_type[rt] = trace
                a0_by_type[rt] = a0_follow

                # expectation
                Jv_exp += float(b.get(rt, 0.0)) * float(Jv_sum)

            if best is None or Jv_exp < best[0]:
                best = (Jv_exp, a_v)
                best_trace = traces_by_type
                best_a0_by_type = a0_by_type

        Jv_star, a_v_star = best

        # pick "most likely" follower action for the returned vru_action_0 (for legacy compatibility)
        rt_ml = "Aggressive" if b["Aggressive"] >= b["Conservative"] else "Conservative"
        vru_a0_ml = best_a0_by_type[rt_ml]

        return GameMPCOutput(
            a_vehicle=float(a_v_star.a),
            vru_action_0=vru_a0_ml,
            feasible=True,
            debug={
                "detected": True,
                "belief": b,
                "rider_type_ml": rt_ml,
                "Jv_exp": float(Jv_star),
                "a0_by_type": {k: v.mode for k, v in best_a0_by_type.items()},
                "trace_by_type": best_trace,
            }
        )
