"""interaction/controllers/mpc_1d_stackelberg.py

Baseline 1D Stackelberg MPC (current implementation) refactored to use the
shared InteractionOracle.

Leader (ego) enumerates candidate *constant* longitudinal accelerations.
Follower (VRU) computes a best-response action sequence under each rider type.
Leader chooses the action minimizing expected vehicle cost under belief.

This file is Step-5 scaffolding: it preserves your current behavior while
establishing a clean I/O contract shared by other controllers.

DESIGN APPROXIMATIONS
---------------------

!! [M1] Constant-action plan — NOT a trajectory MPC !!
    The ego plan passed to the oracle is always a length-1 EgoPlan: the same
    scalar acceleration a_v is applied unchanged for all H steps. The search
    is therefore over constant-acceleration *primitives*, not over action
    sequences. At H=20 steps (2 s at dt=0.1 s), extreme candidates (e.g.
    -5 m/s² braking) produce unrealistic long-tail vehicle trajectories —
    the vehicle may stop or overshoot within the horizon while the oracle
    continues integrating at that same deceleration. Both the VRU
    best-response and the accumulated vehicle cost are computed against these
    constant-action trajectories. This is a deliberate efficiency trade-off;
    the label "MPC" refers to the receding-horizon replanning structure (one
    action is applied, then the optimisation is re-run), NOT to the
    optimisation of a full action sequence over the horizon.

!! [M2] Myopic (greedy) VRU best-response — NOT full-horizon VRU DP !!
    Inside the oracle, at each horizon step k the VRU picks
        a_s* = argmin_a_s  J_vru_stage(a_v, a_s, state_k)
    using only the current-step stage cost. The VRU does NOT solve a
    horizon-length dynamic programme; it has no look-ahead beyond the
    present step. This means the VRU cannot trade off "yield now for safety
    later" against "go now and risk a future collision." The ego leader
    therefore anticipates a stage-cost-greedy VRU, which may underestimate
    VRU boldness early in the horizon when crossing is still geometrically
    feasible without immediate collision. A full-horizon Stackelberg
    formulation would require backward induction over H steps from the
    VRU's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from interaction.oracle import InteractionOracle, EgoPlan
from interaction.controllers.base import EgoController, EgoControlOutput

from interaction.game.dynamics import VehicleAction, VRUAction
from interaction.game.payoff import PayoffWeights
from interaction.game.game_mpc import rider_weights


@dataclass
class MPC1DConfig:
    horizon_steps: int = 20
    # Step-5 confirmed design:
    # - If detected=False, ego should maintain set speed (cruise PID), and we do NOT
    #   run Stackelberg anticipation. This prevents early braking.
    vru_react_when_undetected: bool = False

    # Cruise PID (used when detected=False)
    cruise_kp: float = 0.80
    cruise_ki: float = 0.10
    cruise_kd: float = 0.00
    cruise_i_clamp: float = 10.0

    # Optional debug: include belief-weighted cost breakdown over candidates.
    # Turn this on only when validating logic; keep off for large sweeps.
    debug_cost_breakdown: bool = False
    include_trace_when_debug: bool = False


class MPC1DStackelbergController(EgoController):
    name = "mpc1d"

    def __init__(
        self,
        *,
        oracle: InteractionOracle,
        vehicle_actions: List[VehicleAction],
        weights_vehicle: PayoffWeights,
        config: MPC1DConfig | None = None,
        verbose: bool = False,
    ):
        self.oracle = oracle
        self.AV = list(vehicle_actions)
        self.wv = weights_vehicle
        self.cfg = config or MPC1DConfig(horizon_steps=oracle.H)
        self.verbose = bool(verbose)

        # Cruise PID memory (receding-horizon controllers are stateful)
        self._cruise_int = 0.0
        self._cruise_prev_err = None
        self._last_reset_t = None

        # Track previous detected flag to reset PID derivative on True→False transition.
        # Prevents a derivative spike on the first cruise step after a Stackelberg phase.
        self._prev_detected: bool = False

        if not self.AV:
            raise ValueError("vehicle_actions cannot be empty")

    def _belief_probs(self, belief: Any) -> Dict[str, float]:
        if belief is None:
            raise ValueError("belief is mandatory (Step 3)")

        if hasattr(belief, "probs"):
            b = dict(belief.probs())
        else:
            b = {
                "Aggressive": float(belief.get("Aggressive", 0.5)),
                "Conservative": float(belief.get("Conservative", 0.5)),
            }
        # normalize defensively
        s = float(b.get("Aggressive", 0.0) + b.get("Conservative", 0.0))
        if s <= 1e-9:
            return {"Aggressive": 0.5, "Conservative": 0.5}
        return {"Aggressive": float(b.get("Aggressive", 0.0)) / s,
                "Conservative": float(b.get("Conservative", 0.0)) / s}

    def _reset_cruise_pid_if_new_episode(self, veh0) -> None:
        """Reset integrators when a new episode starts.

        We assume each episode resets veh.t to ~0.
        """
        t = float(getattr(veh0, "t", 0.0))
        dt = float(getattr(veh0, "dt", 0.1))
        if (self._last_reset_t is None) or (t <= 0.5 * dt):
            self._cruise_int = 0.0
            self._cruise_prev_err = None
            self._last_reset_t = t

    def _cruise_pid(self, *, ref_speed: float, v: float, dt: float, a_min: float, a_max: float) -> float:
        """Simple speed-hold PID producing a continuous longitudinal acceleration."""
        err = float(ref_speed) - float(v)
        self._cruise_int += err * float(dt)
        # integral windup clamp
        ic = float(self.cfg.cruise_i_clamp)
        if ic > 0.0:
            self._cruise_int = max(-ic, min(ic, self._cruise_int))

        if self._cruise_prev_err is None:
            derr = 0.0
        else:
            derr = (err - float(self._cruise_prev_err)) / max(float(dt), 1e-6)
        self._cruise_prev_err = err

        a = (
            float(self.cfg.cruise_kp) * err
            + float(self.cfg.cruise_ki) * float(self._cruise_int)
            + float(self.cfg.cruise_kd) * derr
        )
        return max(float(a_min), min(float(a_max), float(a)))

    def plan(
        self,
        *,
        veh0,
        vru0,
        delta0: float,
        ref_speed: float,
        detected: bool,
        belief: Any,
        scenario: str,
        **kwargs,
    ) -> EgoControlOutput:
        b = self._belief_probs(belief)

        type_list = ["Aggressive", "Conservative"]

        # Allow per-call override
        vru_react_when_undetected = bool(kwargs.get("vru_react_when_undetected", self.cfg.vru_react_when_undetected))

        # Optional debug knobs (off by default)
        debug_costs = bool(kwargs.get("debug_costs", self.cfg.debug_cost_breakdown or self.verbose))
        include_trace = bool(kwargs.get("include_trace", self.cfg.include_trace_when_debug))

        # Step-5 confirmed behavior:
        # If not detected, ego maintains ref speed (PID cruise). No Stackelberg anticipation.
        if not bool(detected):
            # Reset derivative term on True→False transition to prevent a spike caused
            # by the large error jump between the last cruise step (before detection) and
            # now (after the Stackelberg phase, where speed may have changed significantly).
            # The integral is intentionally preserved to avoid set-speed undershoot.
            if self._prev_detected:
                self._cruise_prev_err = None
            self._prev_detected = False
            self._reset_cruise_pid_if_new_episode(veh0)
            p = getattr(veh0, "params", None)
            a_min = float(getattr(p, "a_min", -6.0)) if p is not None else -6.0
            a_max = float(getattr(p, "a_max", 6.0)) if p is not None else 6.0
            a_cmd = self._cruise_pid(
                ref_speed=float(ref_speed),
                v=float(getattr(veh0, "v", 0.0)),
                dt=float(getattr(veh0, "dt", 0.1)),
                a_min=a_min,
                a_max=a_max,
            )
            return EgoControlOutput(
                a_cmd=float(a_cmd),
                delta_cmd=float(delta0),
                feasible=True,
                debug={
                    "detected": False,
                    "policy": "cruise_pid",
                    "belief": b,
                    "scenario": str(scenario),
                    "vru_react_when_undetected": bool(vru_react_when_undetected),
                },
            )

        self._prev_detected = True

        best = None
        best_debug = None
        best_rollouts_by_type: Dict[str, Any] | None = None

        # If debugging, keep a table of candidate evaluations
        cand_table = []

        for a_v in self.AV:
            ego_plan = EgoPlan(
                a_seq=[a_v],
                delta_seq=[float(delta0)],
            )

            Jv_exp = 0.0
            rollouts_by_type: Dict[str, Any] = {}
            a0_by_type: Dict[str, Any] = {}
            Jv_by_type: Dict[str, float] = {}
            min_ttc_by_type: Dict[str, float] = {}

            for rt in type_list:
                ws = rider_weights(rt)
                roll = self.oracle.rollout_best_response(
                    veh0=veh0,
                    vru0=vru0,
                    ego_plan=ego_plan,
                    ref_speed=ref_speed,
                    weights_vehicle=self.wv,
                    weights_vru=ws,
                )

                rollouts_by_type[rt] = roll
                a0_by_type[rt] = roll.a0  # keep VRUAction object (sim can select true rider)
                Jv_by_type[rt] = float(roll.Jv_sum)

                # min TTC over horizon (use oracle stage debug)
                if roll.trace:
                    min_ttc = min(float(row.get("ttc", 1e9)) for row in roll.trace)
                else:
                    min_ttc = 1e9
                min_ttc_by_type[rt] = float(min_ttc)

                Jv_exp += float(b.get(rt, 0.0)) * float(roll.Jv_sum)

            if debug_costs:
                cand_row = {
                    "a": float(a_v.a),
                    "Jv_exp": float(Jv_exp),
                    "belief": dict(b),
                    "Jv_by_type": dict(Jv_by_type),
                    "min_ttc_by_type": dict(min_ttc_by_type),
                    "a0_by_type": {k: (v.mode if hasattr(v, "mode") else str(v)) for k, v in a0_by_type.items()},
                }
                cand_table.append(cand_row)

            if (best is None) or (Jv_exp < best[0]):
                best = (float(Jv_exp), a_v)
                best_rollouts_by_type = dict(rollouts_by_type)
                best_debug = {
                    "detected": bool(detected),
                    "belief": b,
                    "scenario": scenario,
                    "Jv_exp": float(Jv_exp),
                    "a0_by_type": a0_by_type,  # VRUAction objects
                    "Jv_by_type": dict(Jv_by_type),
                    "min_ttc_by_type": dict(min_ttc_by_type),
                }

        assert best is not None
        _, a_star = best

        # Attach optional debug after we know the final winner.
        if best_debug is not None:
            if debug_costs:
                best_debug["candidates"] = list(cand_table)
            if include_trace and best_rollouts_by_type is not None:
                best_debug["trace_by_type"] = {rt: best_rollouts_by_type[rt].trace for rt in type_list}

        # for convenience, also report most-likely rider
        rt_ml = "Aggressive" if b["Aggressive"] >= b["Conservative"] else "Conservative"
        if best_debug is not None:
            best_debug["rider_type_ml"] = rt_ml

        return EgoControlOutput(
            a_cmd=float(a_star.a),
            delta_cmd=float(delta0),
            feasible=True,
            debug=best_debug or {},
        )