"""interaction/controllers/mpc_2d.py

Hierarchical 2D Stackelberg-style MPC (MPC2D)
--------------------------------------------

This controller implements a strategic + tactical hierarchy for coupled longitudinal
and lateral control. It is designed for ego-vehicle vs. VRU (e-scooter) interaction.

Key design (robust fixes)
-------------------------
Fix 1/2/3:
- Strategic outputs a hard "contract" (mode + corridor + speed envelope + commitment).
- Strategic updates are gated (slow clock + dwell time + switching margin).
- Detours use reachable transition (time-varying corridor / reference) and longer horizon.

Fix 4:
- Tactical enforces the contract as a HARD corridor constraint (no "detour bound around y_target").

Fix 5:
- Strategic requires a feasibility certificate (cheap tactical dry-run) before accepting a mode.

Fix 6:
- Strategic follower best-response uses an ego rollout generator that matches tactical primitives
  (coarse beam search under the contract). Strategy–tactical mismatch triggers earlier replan.

Fix 7:
- Instrumentation: strategic mode changes, feasibility outcomes, tactical compliance, overrides,
  and mismatch flags are logged in debug outputs and optional JSONL logs.

Road/lane geometry (two-lane road)
----------------------------------
- Lane centers: y = -2 and y = -6
- Road bounds: y in [-8, 0]
- Lane divider at y = -4 (implied by centers)
- Yield point: x_yield = vru.x - 2.0 (safe distance to current VRU x position)

Notes
-----
- The tactical planner is a beam search over discrete (a, delta) candidates.
- When undetected=False (VRU not in FOV): cruise + return-to-home lane with pure pursuit.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Callable

import math
import os
import json

from interaction.controllers.base import EgoController, EgoControlOutput
from interaction.oracle import InteractionOracle, EgoPlan
from interaction.game.dynamics import VehicleAction, step_vehicle
from interaction.game.state import VehState, VRUState, copy_veh, copy_vru
from interaction.game.payoff import ttc as ttc_los, vehicle_radius
from interaction.game.game_mpc import rider_weights


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _smoothstep5(s: float) -> float:
    s = _clamp(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


# =========================
# Contract + Config
# =========================

@dataclass
class MPC2DContract:
    mode: str  # KEEP | YIELD | DETOUR_LEFT | DETOUR_RIGHT
    dt: float
    N: int

    # corridor (hard)
    y_center: List[float]
    y_lo: List[float]
    y_hi: List[float]

    # speed reference / envelope
    v_ref: List[float]
    v_lo: Optional[List[float]] = None
    v_hi: Optional[List[float]] = None

    # YIELD stopline: hard constraint x <= x_max (constant over horizon)
    x_max: Optional[float] = None

    # commitment / hysteresis
    commit_until_t: float = 0.0

    # terminal requirement name (for debug)
    terminal: str = ""


@dataclass
class MPC2DConfig:
    # -----------------
    # Strategic layer
    # -----------------
    strategic_horizon_s: float = 2.0
    strategic_dt: float = 0.5

    # Strategic update gating
    strategic_period_s: float = 1.0
    min_dwell_s: float = 2.0
    switch_margin: float = 0.20
    yield_confirm_s: float = 0.7

    # Modes and lane geometry
    y_targets: Tuple[float, float] = (-2.0, -6.0)
    # Backward-compatible aliases (road bounds)
    y_min: float = -8.0
    y_max: float = 0.0
    y_road_min: float = -8.0
    y_road_max: float = 0.0
    lane_margin: float = 0.50  # keep off the paint/edge

    # Yield definition
    yield_safe_dist: float = 2.0
    yield_x_margin: float = 0.30
    v_creep: float = 0.8
    v_yield_approach: float = 8.0
    veh_front_overhang: float = 3.0
    a_comfort: float = 2.0
    a_max_brake: float = 5.0
    pass_margin: float = 1.5

    # Detour transition
    detour_T_trans: float = 1.8
    detour_w_start: float = 2.0
    detour_w_end: float = 1.5
    corridor_widen_emergency: float = 0.5  # controlled widening only

    # Tactical horizons (mode-dependent)
    tactical_dt: float = 0.1
    tactical_steps_keep: int = 20     # 2.0 s  [Phase-1: extended from 5 (0.5 s) to cover conflict horizon]
    tactical_steps_detour: int = 18   # 1.8 s
    tactical_steps_yield: int = 20    # 2.0 s  [Fix H: extended to cover VRU crossing event]

    # Candidate discretization (tactical)
    delta_max_deg: float = 25.0
    delta_candidates_deg: Tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    beam_width: int = 40

    # Dry-run feasibility certificate (strategic)
    dry_beam_width: int = 12
    dry_delta_candidates_deg: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)
    dry_steps_keep: int = 4
    dry_steps_detour: int = 10
    dry_steps_yield: int = 14  # [Fix H: matched to extended tactical horizon]
    dry_max_jerk: float = 12.0          # m/s^3
    dry_max_deltarate: float = 2.2      # rad/s
    dry_max_sign_flips: int = 1         # steering sign flips allowed

    # TTC thresholds (for emergency overrides and hysteresis)
    ttc_enter: float = 4.0 #3
    ttc_exit: float = 5.0 #4
    ttc_hard: float = 0.5 # 1.5
    ttc_crit: float = 0.5 # 1.5
    w_ttc: float = 180.0

    # Tracking / comfort
    w_track_x: float = 0.8
    w_track_y: float = 6.0
    w_speed: float = 2.0
    w_acc: float = 0.15
    w_delta: float = 0.06
    w_jerk: float = 0.06
    w_deltarate: float = 0.06
    track_scale_emergency: float = 0.20

    # Footprint-aware proximity (clearance)
    clearance_margin: float = 1.25 # 0.75
    w_clearance: float = 120.0

    # Terminal stabilization
    w_terminal_y: float = 25.0
    w_terminal_yaw: float = 8.0

    # Strategic scoring
    risk_sigma: float = 1.5
    w_strat_risk: float = 1.0
    w_strat_lane_change: float = 0.15
    w_strat_switch: float = 0.10
    w_strat_yield: float = 0.25  # mild penalty to avoid unnecessary stops

    # Cruise control when undetected
    cruise_kp: float = 1.0
    cruise_a_min: float = -3.0
    cruise_a_max: float = 2.0

    # Lane-return controller (when undetected)
    return_lookahead: float = 10.0

    # Detour speed cap during lateral shift
    v_detour_cap: float = 7.0

    # Emergency evasion
    evasion_delta_deg: float = 20.0

    # Strategy–tactical mismatch handling
    mismatch_delta_deg: float = 6.0
    mismatch_a: float = 0.8
    mismatch_replan_delay_s: float = 0.1

    # Optional logging
    log_dir: Optional[str] = None  # if set, writes JSONL logs



    def __post_init__(self) -> None:
        # If user provides legacy y_min/y_max, mirror into y_road_min/y_road_max.
        # If user provides y_road_min/y_road_max only, adopt into y_min/y_max.
        default_min, default_max = -8.0, 0.0
        if (float(self.y_min) == default_min and float(self.y_max) == default_max) and (
            float(self.y_road_min) != default_min or float(self.y_road_max) != default_max
        ):
            self.y_min = float(self.y_road_min)
            self.y_max = float(self.y_road_max)
        else:
            self.y_road_min = float(self.y_min)
            self.y_road_max = float(self.y_max)
class MPC2DController(EgoController):
    name = "mpc2d"

    def __init__(
        self,
        *,
        oracle: InteractionOracle,
        vehicle_actions: List[VehicleAction],
        weights_vehicle,
        config: Optional[MPC2DConfig] = None,
        verbose: bool = False,
    ):
        self.oracle = oracle
        self.AV = list(vehicle_actions)
        self.wv = weights_vehicle
        self.cfg: MPC2DConfig = config or MPC2DConfig()
        self.verbose = bool(verbose)

        if not self.AV:
            raise ValueError("vehicle_actions cannot be empty")

        # Ensure 0 accel exists
        if all(abs(float(a.a)) > 1e-9 for a in self.AV):
            self.AV.append(VehicleAction(a=0.0))

        # coarse strategic oracle
        Ns = max(1, int(round(self.cfg.strategic_horizon_s / max(self.cfg.strategic_dt, 1e-6))))
        self._Ns = Ns
        self.oracle_strat = InteractionOracle(horizon_steps=Ns, vru_actions=list(self.oracle.AS))
        self.oracle_1 = InteractionOracle(horizon_steps=1, vru_actions=list(self.oracle.AS))

        # persistent state
        self._y_home: Optional[float] = None
        self._active_contract: Optional[MPC2DContract] = None
        self._active_vru_pred_by_type: Optional[Dict[str, Any]] = None
        self._active_strat_J: Optional[float] = None
        self._active_assumed_first: Optional[Tuple[float, float]] = None  # (a0, delta0)
        self._next_strat_time: float = -1e9
        self._yield_clear_accum: float = 0.0

        # debug logs in-memory (also can write to disk)
        self._events: List[Dict[str, Any]] = []
        self._tick_hist: List[Dict[str, Any]] = []

        if self.cfg.log_dir:
            os.makedirs(self.cfg.log_dir, exist_ok=False)
            self._events_path = os.path.join(self.cfg.log_dir, "mpc2d_events.jsonl")
            self._ticks_path = os.path.join(self.cfg.log_dir, "mpc2d_ticks.jsonl")
        else:
            self._events_path = None
            self._ticks_path = None

    # -----------------------------
    # Main API
    # -----------------------------
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
        cfg = self.cfg
        debug_costs = bool(kwargs.get("debug_costs", False))
        b = self._belief_probs(belief)

        # Save "home" lane once (episode start)
        if self._y_home is None:
            self._y_home = float(veh0.y)

        t_now = float(getattr(veh0, "t", 0.0))

        # 0) When undetected: cruise + return to home lane
        if not bool(detected):
            a_cmd = self._cruise_accel(veh0=veh0, ref_speed=ref_speed)
            delta_cmd = self._lane_return_delta(veh0=veh0, y_ref=float(self._y_home))
            dbg = {
                "policy": "cruise_return",
                "detected": False,
                "belief": b,
                "scenario": scenario,
                "y_home": float(self._y_home),
                "contract": None,
            }
            # self._append_tick(t_now, veh0, vru0, dbg, override=None)
            return EgoControlOutput(a_cmd=float(a_cmd), delta_cmd=float(delta_cmd), feasible=True, debug=dbg)

        # 1) Strategic update gating (slow clock + triggers)
        need_strat = (self._active_contract is None) or (t_now >= float(self._next_strat_time) - 1e-9)
        # tactical–strategy mismatch may request earlier update
        mismatch_info: Optional[Dict[str, Any]] = None

        if need_strat:
            self._active_contract, self._active_strat_J, self._active_vru_pred_by_type, strat_dbg = self._select_contract(
                veh0=veh0, vru0=vru0, delta0=delta0, ref_speed=ref_speed, belief=b, t_now=t_now,
                scenario=scenario,  # [Phase-3A] forward scenario for mode filtering
            )
            self._next_strat_time = t_now + float(cfg.strategic_period_s)
        else:
            strat_dbg = {"kept_contract": True}

        assert self._active_contract is not None
        assert self._active_vru_pred_by_type is not None

        contract = self._active_contract

        # 2) Tactical execution inside contract
        _v0_yield = float(veh0.v)
        _a_hot_yield = min(float(a.a) for a in self.AV) if self.AV else -5.0
        tact = self._tactical_beam_search_contract(
            veh0=veh0,
            vru0=vru0,
            contract=contract,
            belief=b,
            delta0=float(delta0),
            vru_pred_by_type=self._active_vru_pred_by_type,
            debug_costs=debug_costs,
            a_prev_override=_a_hot_yield if (contract.mode == "YIELD" and _v0_yield > float(cfg.v_yield_approach)) else None,
        )

        # Analytical stopping-distance override (Fix F)
        if contract.mode == "YIELD" and contract.x_max is not None:
            _x_stop_plan = float(veh0.x) + (_v0_yield * _v0_yield) / (2.0 * max(abs(_a_hot_yield), 1e-3))
            if float(veh0.x) < float(contract.x_max) and _x_stop_plan > float(contract.x_max):
                tact = {"feasible": False, "a_cmd": _a_hot_yield, "delta_cmd": 0.0}

        override: Optional[Dict[str, Any]] = None

        if not bool(tact.get("feasible", True)):
            # infeasible under current contract -> force immediate YIELD replan, else emergency brake
            # self._log_event({"t": t_now, "type": "override", "trigger": "tactical_infeasible", "mode": contract.mode})
            override = {"trigger": "tactical_infeasible", "mode": contract.mode}

            yield_contract = self._build_contract("YIELD", veh0=veh0, vru0=vru0, ref_speed=ref_speed, t_now=t_now)
            # Generate VRU predictions for YIELD (aligned ego rollout)
            yield_pred = self._predict_vru_by_type_for_contract(
                veh0=veh0, vru0=vru0, ref_speed=ref_speed, contract=yield_contract, belief=b
            )
            tact2 = self._tactical_beam_search_contract(
                veh0=veh0,
                vru0=vru0,
                contract=yield_contract,
                belief=b,
                delta0=float(delta0),
                vru_pred_by_type=yield_pred["vru_pred_by_type"],
                debug_costs=debug_costs,
                a_prev_override=_a_hot_yield if _v0_yield > float(cfg.v_yield_approach) else None,
            )
            if bool(tact2.get("feasible", False)):
                contract = yield_contract
                tact = tact2
                self._active_contract = yield_contract
                self._active_vru_pred_by_type = yield_pred["vru_pred_by_type"]
                self._active_strat_J = float("nan")
                self._active_assumed_first = tuple(yield_pred.get("assumed_first", (0.0, 0.0)))
                self._next_strat_time = t_now + float(cfg.mismatch_replan_delay_s)
                override = {"trigger": "tactical_infeasible->yield", "mode": "YIELD"}
            else:
                # # emergency brake straight
                # a_cmd = min(float(a.a) for a in self.AV) if self.AV else -3.0
                # dbg = {
                #     "policy": "mpc2d_emergency_brake",
                #     "detected": True,
                #     "belief": b,
                #     "scenario": scenario,
                #     "contract": self._contract_brief(yield_contract),
                #     "override": {"trigger": "tactical_infeasible->yield_infeasible", "action": "brake_straight"},
                #     "strat": strat_dbg,
                #     "tact": tact2,
                # }
                # self._append_tick(t_now, veh0, vru0, dbg, override=dbg["override"])
                # return EgoControlOutput(a_cmd=float(a_cmd), delta_cmd=0.0, feasible=False, debug=dbg)


                # ----------------------------------------------------------------------
                # EMERGENCY EVASION PATCH (when BOTH: current contract infeasible AND YIELD infeasible)
                #
                # Context:
                # - We already tried to recover from tactical infeasibility by forcing a YIELD contract.
                # - If YIELD is still infeasible, it usually means:
                #     (1) we cannot satisfy the YIELD stopline constraint (x_max) with the available braking,
                #         given current speed and remaining distance, AND/OR
                #     (2) the corridor + collision constraints leave no admissible (a, delta) sequence within horizon.
                #
                # In that case, "brake straight" alone is often NOT enough (your collision example),
                # because the ego will remain in (or close to) the same lane tube while the VRU is
                # predicted to keep crossing into that tube.
                #
                # Emergency Evasion policy:
                # - We keep maximum braking (still the best longitudinal action).
                # - BUT we add a small steering bias AWAY from the VRU laterally:
                #     steer to the opposite side of the VRU (relative to ego's current y).
                # - This is intentionally an OVERRIDE behavior: it is NOT trying to satisfy the YIELD contract,
                #   because the YIELD contract is already proven infeasible.
                #
                # Safety notes:
                # - Steering magnitude should be "small but meaningful": enough to create lateral separation,
                #   not so large that we cause aggressive lateral instability.
                # - The sign convention depends on your vehicle model:
                #     the code below assumes positive delta tends to move y toward 0 (up),
                #     and negative delta tends to move y toward -8 (down).
                #   If your model uses the opposite convention, flip the sign logic here.
                #
                # Debug requirements:
                # - We log this as a first-class override event (so it is never silent).
                # - We force an immediate strategic update on the next tick, because the hierarchy
                #   is now operating in an emergency override regime.
                # ----------------------------------------------------------------------

                # maximum available braking (same as before)
                a_cmd = min(float(a.a) for a in self.AV) if self.AV else -3.0

                # Determine the evasion direction: steer away from the VRU.
                # If VRU is above ego (higher y), evade downward (more negative y).
                # If VRU is below ego (lower y), evade upward (toward 0).
                y_ego = float(veh0.y)
                y_vru = float(vru0.y)

                evade_down = (y_vru >= y_ego)  # steer away from VRU's current y position
                # Pick a small evasion steering angle.
                # Suggested knob: cfg.evasion_delta_deg (add to MPC2DConfig), default around 6 to 10 degrees.
                # If you don't want to add a new config knob yet, you can hardcode 8.0 here.
                evasion_deg = float(getattr(cfg, "evasion_delta_deg", 8.0))

                delta_evasion = math.radians(evasion_deg) * (-1.0 if evade_down else +1.0)

                # Always clamp by the global steering bound.
                delta_max = math.radians(float(cfg.delta_max_deg))
                delta_cmd = float(max(-delta_max, min(delta_max, delta_evasion)))

                # Record override so the log clearly indicates why we violated the nominal hierarchy.
                override = {
                    "trigger": "tactical_infeasible->yield_infeasible",
                    "action": "emergency_brake_and_evasion",
                    "evasion_side": "down" if evade_down else "up",
                    "delta_cmd": float(delta_cmd),
                    "a_cmd": float(a_cmd),
                }

                # Force strategy to re-evaluate immediately on next tick (do not wait for strategic_period_s),
                # because we are in an emergency override regime and the VRU interaction is evolving fast.
                self._next_strat_time = min(self._next_strat_time, t_now)

                # Log and return (feasible=False because no feasible plan existed within the contract constraints).
                dbg = {
                    "policy": "mpc2d_emergency_evasion",
                    "detected": True,
                    "belief": b,
                    "scenario": scenario,
                    "contract": self._contract_brief(yield_contract),
                    "override": override,
                    "strat": strat_dbg,
                    "tact": tact2,
                }

                # self._log_event({"t": t_now, "type": "override", **override})
                # self._append_tick(t_now, veh0, vru0, dbg, override=override)
                return EgoControlOutput(a_cmd=float(a_cmd), delta_cmd=float(delta_cmd), feasible=False, debug=dbg)


        a_cmd = float(tact["a_cmd"])
        delta_cmd = float(tact["delta_cmd"])

        # [Phase-3B] Reactive safety override: if tactical TTC is still critically low, max-brake.
        # tact["ttc_min_b"] is already computed by the beam search at no extra cost.
        if bool(tact.get("feasible", True)) and override is None:
            if float(tact.get("ttc_min_b", float("inf"))) < float(cfg.ttc_hard):
                a_cmd = min(float(a.a) for a in self.AV)
                delta_cmd = 0.0
                override = {
                    "trigger": "reactive_safety_override",
                    "ttc": float(tact.get("ttc_min_b", 0.0)),
                }

        # 3) Strategy–tactical mismatch detection (Fix 6)
        assumed = self._active_assumed_first
        if assumed is not None:
            a_assumed, d_assumed = float(assumed[0]), float(assumed[1])
            d_err_deg = abs(math.degrees(float(delta_cmd) - float(d_assumed)))
            a_err = abs(float(a_cmd) - float(a_assumed))
            if (d_err_deg > float(cfg.mismatch_delta_deg)) or (a_err > float(cfg.mismatch_a)):
                mismatch_info = {"a_err": a_err, "d_err_deg": d_err_deg, "assumed": [a_assumed, d_assumed]}
                # self._log_event({"t": t_now, "type": "mismatch", "mode": contract.mode, **mismatch_info})
                self._next_strat_time = min(self._next_strat_time, t_now + float(cfg.mismatch_replan_delay_s))

        # [Phase-2C] YIELD exit: accumulate time with TTC > ttc_exit; force replan when clear
        if contract.mode == "YIELD":
            step_dt = float(getattr(veh0, "dt", float(cfg.tactical_dt)))
            ttc_now = float(tact.get("ttc_min_b", 0.0))
            if ttc_now > float(cfg.ttc_exit):
                self._yield_clear_accum += step_dt
            else:
                self._yield_clear_accum = 0.0
            if self._yield_clear_accum >= float(cfg.yield_confirm_s):
                self._yield_clear_accum = 0.0
                self._next_strat_time = min(self._next_strat_time, t_now)  # force immediate replan
        else:
            self._yield_clear_accum = 0.0

        # 4) First-step VRU best response for logging/analysis
        # a0_by_type = self._compute_a0_by_type(veh0=veh0, vru0=vru0, ref_speed=ref_speed, a_cmd=a_cmd, delta_cmd=delta_cmd)

        dbg: Dict[str, Any] = {
            "policy": "mpc2d_hier_contract",
            "detected": True,
            "belief": b,
            "scenario": scenario,
            "contract": self._contract_brief(contract),
            "strat": strat_dbg,
            "ttc_min_b": tact.get("ttc_min_b"),
            "ttc_trigger": tact.get("ttc_trigger"),
            "beam_log": tact.get("beam_log"),
            "corridor": tact.get("corridor"),  # time-indexed corridor used by tactical
            # "a0_by_type": a0_by_type,
            "y_home": float(self._y_home),
            "override": override,
            "mismatch": mismatch_info,
        }

        # self._append_tick(t_now, veh0, vru0, dbg, override=override)
        return EgoControlOutput(a_cmd=a_cmd, delta_cmd=float(delta_cmd), feasible=bool(tact.get("feasible", True)), debug=dbg)

    # =============================
    # Contract builder
    # =============================
    def _lane_bounds(self, y_center: float) -> Tuple[float, float]:
        """Return ideal lane bounds (without margin) given lane center."""
        cfg = self.cfg
        y_c = float(y_center)
        # implied lane width 4m (centers differ by 4); divider at -4
        if y_c > -4.0:
            lo, hi = -4.0, -1.0
        else:
            lo, hi = -7.0, -4.0
        lo = max(float(cfg.y_road_min), lo)
        hi = min(float(cfg.y_road_max), hi)
        return float(lo), float(hi)

    def _current_lane_center(self, y: float) -> float:
        return -2.0 if float(y) > -4.0 else -6.0

    def _nominal_lane_center(self) -> float:
        cfg = self.cfg
        y_home = float(self._y_home if self._y_home is not None else -2.0)
        return float(min(cfg.y_targets, key=lambda yt: abs(float(yt) - y_home)))

    def _build_contract(
        self,
        mode: str,
        *,
        veh0: VehState,
        vru0: VRUState,
        ref_speed: float,
        t_now: float,
        dt_override: Optional[float] = None,
        N_override: Optional[int] = None,
    ) -> MPC2DContract:
        cfg = self.cfg
        mode = str(mode)

        dt = float(dt_override) if dt_override is not None else float(cfg.tactical_dt)

        if mode == "DETOUR_LEFT" or mode == "DETOUR_RIGHT":
            N = int(N_override) if N_override is not None else int(cfg.tactical_steps_detour)
        elif mode == "YIELD":
            N = int(N_override) if N_override is not None else int(cfg.tactical_steps_yield)
        else:
            N = int(N_override) if N_override is not None else int(cfg.tactical_steps_keep)

        y0 = float(veh0.y)
        v0 = float(veh0.v)

        # Corridor target definition
        if mode == "KEEP":
            y_goal = self._nominal_lane_center()
            # If already far from nominal, create a short transition back (Fix 3 style)
            T_trans = max(0.8, float(cfg.detour_T_trans) * 0.8)
            terminal = "within_nominal_lane"
            v_ref = [min(float(ref_speed), float(ref_speed)) for _ in range(N)]
            lo_lane, hi_lane = self._lane_bounds(y_goal)
        elif mode == "YIELD":
            y_goal = self._current_lane_center(y0)  # stay in current lane while braking
            T_trans = 0.0
            terminal = "stop_before_yield_x"
            lo_lane, hi_lane = self._lane_bounds(y_goal)
            # [Phase-4] Predict conflict x from VRU lateral velocity.
            # t_cross: estimated time for VRU to reach ego lane center y.
            # Falls back to current vru.x if VRU is not clearly approaching.
            _ego_lane_y = float(self._current_lane_center(y0))
            _vy_vru = float(getattr(vru0, "vy", 0.0))
            _delta_y = _ego_lane_y - float(vru0.y)
            _vy_thresh = 0.15  # m/s: below this treat VRU as laterally stationary
            if abs(_vy_vru) >= _vy_thresh:
                _t_cross = _delta_y / _vy_vru
            else:
                _t_cross = -1.0
            if 0.0 < _t_cross <= 10.0:  # approaching and within horizon cap
                _vx_vru = float(getattr(vru0, "vx", 0.0))
                _x_conflict = float(vru0.x) + _vx_vru * _t_cross
            else:
                _x_conflict = float(vru0.x)  # fallback: stationary, moving away, or capped
            # yield x constraint
            x_yield = _x_conflict - float(cfg.yield_safe_dist)
            x_yield = max(x_yield, float(veh0.x))  # don't put stopline behind ego
            x_max = x_yield - float(cfg.yield_x_margin) - float(cfg.veh_front_overhang)
            # speed reference to stop/creep
            d = max(0.0, x_max - float(veh0.x))
            eps = 1e-3
            if v0 > float(cfg.v_yield_approach):
                # Fast approach: use max braking so v_ref is steep enough
                # to guide the tactical beam toward hard braking (avoid jerk-wall stall).
                a_cmd = -float(cfg.a_max_brake)
            else:
                a_req = - (v0 * v0) / (2.0 * max(d, eps)) if d > 0.0 else -float(cfg.a_max_brake)
                a_cmd = _clamp(a_req, -float(cfg.a_max_brake), -float(cfg.a_comfort))
            v_ref = []
            for k in range(N):
                vk = max(float(cfg.v_creep), v0 + a_cmd * (k + 1) * dt)
                v_ref.append(float(vk))
        elif mode == "DETOUR_LEFT":
            y_goal = -6.0
            T_trans = float(cfg.detour_T_trans)
            terminal = "establish_left_lane"
            v_ref = []
            for k in range(N):
                tau = (k + 1) * dt
                if tau <= T_trans:
                    v_ref.append(float(min(ref_speed, float(cfg.v_detour_cap))))
                else:
                    v_ref.append(float(ref_speed))
            lo_lane, hi_lane = self._lane_bounds(y_goal)
        elif mode == "DETOUR_RIGHT":
            y_goal = -2.0
            T_trans = float(cfg.detour_T_trans)
            terminal = "establish_right_lane"
            v_ref = []
            for k in range(N):
                tau = (k + 1) * dt
                if tau <= T_trans:
                    v_ref.append(float(min(ref_speed, float(cfg.v_detour_cap))))
                else:
                    v_ref.append(float(ref_speed))
            lo_lane, hi_lane = self._lane_bounds(y_goal)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build time-indexed y_center and corridor bounds (Fix 4)
        y_center: List[float] = []
        y_lo: List[float] = []
        y_hi: List[float] = []

        margin = float(cfg.lane_margin)
        y_road_min = float(cfg.y_road_min)
        y_road_max = float(cfg.y_road_max)

        for k in range(N):
            tau = (k + 1) * dt
            if T_trans > 1e-6:
                s = _smoothstep5(tau / T_trans)
            else:
                s = 1.0

            yc = y0 + s * (float(y_goal) - y0)
            y_center.append(float(yc))

            if mode == "YIELD":
                # stay-in-lane tube
                lo, hi = lo_lane + margin, hi_lane - margin
            elif mode == "KEEP" and abs(y0 - float(y_goal)) <= 0.5:
                # already near nominal lane: strict lane tube
                lo, hi = lo_lane + margin, hi_lane - margin
            else:
                # transition-aware corridor: wide early, tight late
                w = float(cfg.detour_w_start) + s * (float(cfg.detour_w_end) - float(cfg.detour_w_start))
                lo, hi = yc - w, yc + w

            lo = _clamp(float(lo), y_road_min + margin, y_road_max - margin)
            hi = _clamp(float(hi), y_road_min + margin, y_road_max - margin)
            if lo > hi:
                lo, hi = hi, lo
            y_lo.append(float(lo))
            y_hi.append(float(hi))

        x_max_val = None
        if mode == "YIELD":
            x_yield = _x_conflict - float(cfg.yield_safe_dist)  # [Phase-4] use conflict-x
            x_yield = max(x_yield, float(veh0.x))
            x_max_val = float(x_yield - float(cfg.yield_x_margin) - float(cfg.veh_front_overhang))

        commit_until_t = float(t_now)
        if mode in ("DETOUR_LEFT", "DETOUR_RIGHT"):
            commit_until_t = float(t_now + float(cfg.min_dwell_s))

        return MPC2DContract(
            mode=mode,
            dt=dt,
            N=N,
            y_center=y_center,
            y_lo=y_lo,
            y_hi=y_hi,
            v_ref=v_ref,
            x_max=x_max_val,
            commit_until_t=commit_until_t,
            terminal=terminal,
        )

    def _contract_brief(self, c: MPC2DContract) -> Dict[str, Any]:
        return {
            "mode": c.mode,
            "dt": c.dt,
            "N": c.N,
            "commit_until_t": c.commit_until_t,
            "x_max": c.x_max,
            "terminal": c.terminal,
            "y_center0": c.y_center[0] if c.y_center else None,
            "y_centerN": c.y_center[-1] if c.y_center else None,
            "v_ref0": c.v_ref[0] if c.v_ref else None,
            "v_refN": c.v_ref[-1] if c.v_ref else None,
        }

    # =============================
    # Strategic selection with feasibility certificate
    # =============================
    def _select_contract(
        self,
        *,
        veh0: VehState,
        vru0: VRUState,
        delta0: float,
        ref_speed: float,
        belief: Dict[str, float],
        t_now: float,
        scenario: str = "",  # [Phase-3A] used for mode filtering
    ) -> Tuple[MPC2DContract, float, Dict[str, Any], Dict[str, Any]]:
        cfg = self.cfg

        active = self._active_contract
        active_mode = active.mode if active is not None else None

        # Dwell time: do not switch detour modes early unless hard-safety trigger
        if active is not None and (t_now < float(active.commit_until_t) - 1e-9):
            # still run a feasibility check on active; if infeasible -> will override in plan()
            dbg = {"kept_due_to_dwell": True, "active_mode": active_mode}
            return active, float(self._active_strat_J or 0.0), dict(self._active_vru_pred_by_type or {}), dbg

        # [Phase-3A] Scenario-aware mode filtering: DETOUR only valid in intersection.
        # In straight_road, lateral lane changes into oncoming traffic are unsafe/infeasible.
        _all_modes = ["KEEP", "YIELD", "DETOUR_LEFT", "DETOUR_RIGHT"]
        _straight_modes = ["KEEP", "YIELD"]
        modes = _all_modes if scenario in ("intersection", "") else _straight_modes
        # Warm-start: evaluate active mode first
        if active_mode in modes:
            modes = [active_mode] + [m for m in modes if m != active_mode]

        considered: List[Dict[str, Any]] = []
        rejected: Dict[str, Any] = {}

        best: Optional[Tuple[MPC2DContract, float, Dict[str, Any], Tuple[float, float]]] = None
        active_eval: Optional[Tuple[float, float]] = None  # (J_active, ttc_active)

        for mode in modes:
            contract = self._build_contract(mode, veh0=veh0, vru0=vru0, ref_speed=ref_speed, t_now=t_now)

            # Build strategic-aligned ego rollout (Fix 6): coarse beam under this contract
            pred = self._predict_vru_by_type_for_contract(
                veh0=veh0, vru0=vru0, ref_speed=ref_speed, contract=contract, belief=belief
            )
            vru_pred_by_type = pred["vru_pred_by_type"]
            assumed_first = tuple(pred.get("assumed_first", (0.0, 0.0)))
            ttc_pred = float(pred.get("ttc_min_b", float("inf")))  # [Phase-2B] oracle-rollout TTC

            # Compute strategic risk score (occupancy kernel) using oracle rollouts already in pred
            J_risk = float(pred.get("J_risk_b", 0.0))
            lane_cost = 0.0
            y_nom = float(self._nominal_lane_center())
            if mode == "KEEP":
                lane_cost = 0.0
            elif mode in ("DETOUR_LEFT", "DETOUR_RIGHT"):
                y_goal = -6.0 if mode == "DETOUR_LEFT" else -2.0
                lane_cost = (float(y_goal) - y_nom) ** 2
            elif mode == "YIELD":
                lane_cost = float(cfg.w_strat_yield)  # mild penalty, not squared

            J = float(cfg.w_strat_risk) * J_risk + float(cfg.w_strat_lane_change) * float(lane_cost)

            if active_mode is not None and mode != active_mode:
                J += float(cfg.w_strat_switch)

            # Feasibility certificate (Fix 5)
            feas = self._tactical_feasibility_certificate(
                veh0=veh0,
                vru0=vru0,
                contract=contract,
                belief=belief,
                delta0=float(delta0),
                vru_pred_by_type=vru_pred_by_type,
            )

            rec = {
                "mode": mode,
                "J": float(J),
                "J_risk": float(J_risk),
                "feasible": bool(feas["feasible"]),
                "reason": feas.get("reason"),
                "ttc_min_b": feas.get("ttc_min_b"),
                "ttc_pred": float(ttc_pred),  # [Phase-2B] oracle-rollout TTC for debug
            }
            considered.append(rec)

            if not bool(feas["feasible"]):
                rejected[mode] = rec
                continue

            ttc_min_b = float(feas.get("ttc_min_b", float("inf")))
            if mode == active_mode:
                active_eval = (float(J), float(ttc_min_b))

            # [Phase-2B] TTC safety gate: reject non-YIELD modes when oracle-rollout TTC < ttc_enter.
            # ttc_enter (4.0 s default) replaces ttc_hard (0.5 s) as the primary early-warning threshold.
            # [Fix G] Temporal pass-through exemption: if ego can clearly pass the conflict zone
            # before the VRU arrives (with margin), do not reject KEEP via the TTC gate.
            _pass_exempt = False
            if mode == "KEEP":
                _vy_g = float(getattr(vru0, "vy", 0.0))
                _vru_x_g = float(vru0.x)
                _ego_lane_y_g = float(self._current_lane_center(float(veh0.y)))
                _delta_y_g = _ego_lane_y_g - float(vru0.y)
                if (float(veh0.v) > 0.5 and float(veh0.x) < _vru_x_g
                        and _vy_g > 0.1 and _delta_y_g > 0.0):
                    _t_ego_pass = (_vru_x_g - float(veh0.x)) / float(veh0.v)
                    _t_vru_cross = _delta_y_g / _vy_g
                    if _t_ego_pass < _t_vru_cross - float(cfg.pass_margin):
                        _pass_exempt = True
            if mode != "YIELD" and ttc_pred < float(cfg.ttc_enter) and not _pass_exempt:
                rejected[mode] = {**rec, "reason": "ttc_enter_gate"}
                continue

            if best is None or float(J) < float(best[1]):
                best = (contract, float(J), vru_pred_by_type, assumed_first)

        # If nothing feasible, fall back to YIELD contract (may still be infeasible tactically -> plan() handles)
        if best is None:
            fallback = self._build_contract("YIELD", veh0=veh0, vru0=vru0, ref_speed=ref_speed, t_now=t_now)
            pred = self._predict_vru_by_type_for_contract(veh0=veh0, vru0=vru0, ref_speed=ref_speed, contract=fallback, belief=belief)
            best = (fallback, float("inf"), pred["vru_pred_by_type"], tuple(pred.get("assumed_first", (0.0, 0.0))))
            # self._log_event({"t": t_now, "type": "strategic_fallback", "reason": "no_feasible_mode"})

        chosen_contract, J_best, pred_best, assumed_first = best

        # Switching margin / hysteresis (Fix 2): keep active if not clearly better
        if active is not None and active_eval is not None and chosen_contract.mode != active.mode:
            J_active, ttc_active = active_eval
            if float(J_best) > float(J_active) - float(cfg.switch_margin):
                chosen_contract = active
                J_best = float(J_active)
                pred_best = dict(self._active_vru_pred_by_type or pred_best)
                assumed_first = tuple(self._active_assumed_first or assumed_first)

        # Log contract change
        if active is None or chosen_contract.mode != active.mode:
            self._log_event({
                "t": t_now,
                "type": "contract_change",
                "from": active.mode if active else None,
                "to": chosen_contract.mode,
                "J": float(J_best),
                "considered": considered,
                "rejected": rejected,
            })

        # Store assumed first action for mismatch checks
        self._active_assumed_first = (float(assumed_first[0]), float(assumed_first[1]))

        dbg = {
            "considered": considered,
            "rejected": rejected,
            "chosen": {"mode": chosen_contract.mode, "J": float(J_best)},
        }
        return chosen_contract, float(J_best), pred_best, dbg

    def _predict_vru_by_type_for_contract(
        self,
        *,
        veh0: VehState,
        vru0: VRUState,
        ref_speed: float,
        contract: MPC2DContract,
        belief: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate an ego rollout aligned with tactical primitives, then get VRU best responses."""
        dt_s = float(self.cfg.strategic_dt)
        Ns = int(self._Ns)

        # coarse contract sampled at strategic dt
        coarse = self._build_contract(
            contract.mode, veh0=veh0, vru0=vru0, ref_speed=ref_speed, t_now=float(getattr(veh0, "t", 0.0)),
            dt_override=dt_s, N_override=Ns
        )
        ego_plan, assumed_first = self._coarse_ego_rollout_under_contract(veh0=veh0, contract=coarse)

        # clone with coarse dt
        vehs0 = copy_veh(veh0)
        vru0s = copy_vru(vru0)
        vehs0.dt = dt_s
        vru0s.dt = dt_s

        pred_by_type: Dict[str, Any] = {}
        risk_b = 0.0
        sigma = max(1e-3, float(self.cfg.risk_sigma))
        ttc_min_b = float("inf")  # [Phase-2A] worst-case TTC across types/steps for strategic gate

        for rt in ("Aggressive", "Conservative"):
            ws = rider_weights(rt)
            roll = self.oracle_strat.rollout_best_response(
                veh0=vehs0,
                vru0=vru0s,
                ego_plan=ego_plan,
                ref_speed=float(ref_speed),
                weights_vehicle=self.wv,
                weights_vru=ws,
            )
            times = [i * dt_s for i in range(len(roll.vru_traj))]
            pred_by_type[rt] = {"traj": roll.vru_traj, "times": times}

            w_b = float(belief.get(rt, 0.0))
            # occupancy risk kernel + TTC extraction
            for i in range(1, min(len(roll.veh_traj), len(roll.vru_traj))):
                dx = float(roll.veh_traj[i].x) - float(roll.vru_traj[i].x)
                dy = float(roll.veh_traj[i].y) - float(roll.vru_traj[i].y)
                d = math.hypot(dx, dy)
                risk = math.exp(-(d * d) / (2.0 * sigma * sigma))
                risk_b += w_b * risk
                # [Phase-2A] track worst-case TTC over all types/steps for strategic TTC gate
                if w_b > 0.0:
                    ttc_k = float(ttc_los(roll.veh_traj[i], roll.vru_traj[i]))
                    ttc_min_b = min(ttc_min_b, ttc_k)

        return {
            "vru_pred_by_type": pred_by_type,
            "J_risk_b": float(risk_b),
            "ttc_min_b": float(ttc_min_b),  # [Phase-2A] oracle-rollout TTC for _select_contract gate
            "assumed_first": assumed_first,
        }

    def _coarse_ego_rollout_under_contract(self, *, veh0: VehState, contract: MPC2DContract) -> Tuple[EgoPlan, Tuple[float, float]]:
        """Coarse beam search (no VRU) to produce an ego plan consistent with the contract (Fix 6)."""
        cfg = self.cfg
        dt = float(contract.dt)
        N = int(contract.N)

        delta_lim = math.radians(float(cfg.delta_max_deg))
        delta_cands = [_clamp(math.radians(d), -delta_lim, delta_lim) for d in cfg.dry_delta_candidates_deg]
        # Use AV action set but include only a few for coarseness
        a_cands = sorted({float(a.a) for a in self.AV})
        # Keep small in coarse rollout
        if len(a_cands) > 5:
            a_cands = [min(a_cands), 0.0, max(a_cands)]
            a_cands = sorted(set(a_cands))

        # precompute x_ref via v_ref
        x_ref = [float(veh0.x)]
        for k in range(N):
            x_ref.append(float(x_ref[-1] + float(contract.v_ref[k]) * dt))

        # Node: (cost, veh, a_prev, d_prev, seq_a, seq_d)
        beam: List[Tuple[float, VehState, float, float, List[float], List[float]]] = [
            (0.0, copy_veh(veh0), float(getattr(veh0, "u_last", 0.0)), 0.0, [], [])
        ]

        for k in range(N):
            new_beam: List[Tuple[float, VehState, float, float, List[float], List[float]]] = []
            xr = float(x_ref[k + 1])
            yr = float(contract.y_center[k])

            for cost, veh, a_prev, d_prev, seq_a, seq_d in beam:
                for a in a_cands:
                    for delta in delta_cands:
                        veh1 = self._step_vehicle_dt(veh, a=float(a), delta=float(delta), dt_override=dt)

                        # hard road bounds
                        if not (float(cfg.y_road_min) - 1e-6 <= float(veh1.y) <= float(cfg.y_road_max) + 1e-6):
                            continue
                        # hard corridor
                        if not (float(contract.y_lo[k]) - 1e-6 <= float(veh1.y) <= float(contract.y_hi[k]) + 1e-6):
                            continue
                        # YIELD stopline
                        if contract.x_max is not None and float(veh1.x) > float(contract.x_max) + 1e-6:
                            continue

                        # simple cost: track (x,y) and v_ref, penalize control changes
                        J = 0.0
                        J += float(cfg.w_track_x) * (float(veh1.x) - xr) ** 2
                        J += float(cfg.w_track_y) * (float(veh1.y) - yr) ** 2
                        J += float(cfg.w_speed) * (float(veh1.v) - float(contract.v_ref[k])) ** 2
                        J += float(cfg.w_acc) * (float(a) ** 2)
                        J += float(cfg.w_delta) * (float(delta) ** 2)

                        jerk = (float(a) - float(a_prev)) / max(dt, 1e-6)
                        drate = (float(delta) - float(d_prev)) / max(dt, 1e-6)
                        J += float(cfg.w_jerk) * (jerk ** 2)
                        J += float(cfg.w_deltarate) * (drate ** 2)

                        new_beam.append((cost + J, veh1, float(a), float(delta), seq_a + [float(a)], seq_d + [float(delta)]))

            new_beam.sort(key=lambda z: z[0])
            beam = new_beam[: int(cfg.dry_beam_width)]

            if not beam:
                break

        if not beam:
            # fallback: straight, no accel
            ego_plan = EgoPlan(a_seq=[VehicleAction(0.0)] * N, delta_seq=[0.0] * N)
            return ego_plan, (0.0, 0.0)

        best = min(beam, key=lambda z: z[0])
        _, _, _, _, seq_a, seq_d = best
        a0 = float(seq_a[0]) if seq_a else 0.0
        d0 = float(seq_d[0]) if seq_d else 0.0

        ego_plan = EgoPlan(a_seq=[VehicleAction(float(a)) for a in seq_a], delta_seq=list(seq_d))
        # ensure correct length
        if len(ego_plan.a_seq) < N:
            ego_plan.a_seq += [VehicleAction(0.0)] * (N - len(ego_plan.a_seq))
        if len(ego_plan.delta_seq) < N:
            ego_plan.delta_seq += [0.0] * (N - len(ego_plan.delta_seq))

        return ego_plan, (a0, d0)

    # =============================
    # Tactical planner under contract (beam search)
    # =============================
    def _tactical_beam_search_contract(
        self,
        *,
        veh0: VehState,
        vru0: VRUState,
        contract: MPC2DContract,
        belief: Dict[str, float],
        delta0: float,
        vru_pred_by_type: Dict[str, Any],
        debug_costs: bool,
        beam_width_override: Optional[int] = None,
        delta_candidates_deg_override: Optional[Tuple[float, ...]] = None,
        stability_caps: Optional[Dict[str, float]] = None,
        a_prev_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        dt = float(contract.dt)
        N = int(contract.N)
        beam_width = int(beam_width_override) if beam_width_override is not None else int(cfg.beam_width)

        # interpolation functions for each type
        pred_interp = {
            rt: self._make_vru_interp(vru_pred_by_type[rt]["traj"], vru_pred_by_type[rt]["times"])
            for rt in ("Aggressive", "Conservative")
        }

        delta_lim = math.radians(float(cfg.delta_max_deg))
        cand_deg = delta_candidates_deg_override if delta_candidates_deg_override is not None else cfg.delta_candidates_deg
        delta_cands = [_clamp(math.radians(d), -delta_lim, delta_lim) for d in cand_deg]
        a_cands = [float(a.a) for a in self.AV]

        # x reference integrates v_ref
        x_ref = [float(veh0.x)]
        for k in range(N):
            x_ref.append(float(x_ref[-1] + float(contract.v_ref[k]) * dt))

        # Node: (cost, veh, a_prev, d_prev, seq_a, seq_d, ttc_min_b, sign_flips)
        beam: List[Tuple[float, VehState, float, float, List[float], List[float], float, int]] = [
            (0.0, copy_veh(veh0), float(a_prev_override) if a_prev_override is not None else float(getattr(veh0, "u_last", 0.0)), float(delta0), [], [], float("inf"), 0)
        ]

        beam_log: List[List[Dict[str, Any]]] = []

        def compute_ttc_and_clearance_cost(veh_state: VehState, tau: float) -> Tuple[float, float, bool]:
            ttc_min = float("inf")
            J_ttc = 0.0
            J_clear = 0.0
            collision = False

            r_veh = float(vehicle_radius(veh_state))

            for rt in ("Aggressive", "Conservative"):
                w_b = float(belief.get(rt, 0.0))
                if w_b <= 0.0:
                    continue
                vru_p = pred_interp[rt](tau)

                dx = float(vru_p.x) - float(veh_state.x)
                dy = float(vru_p.y) - float(veh_state.y)
                dist = math.hypot(dx, dy)
                r_vru = float(getattr(vru_p.params, "R", 0.5))
                clearance = dist - (r_veh + r_vru)

                if clearance <= 0.0:
                    collision = True
                    break

                if clearance < float(cfg.clearance_margin):
                    J_clear += w_b * float(cfg.w_clearance) * (float(cfg.clearance_margin) - clearance) ** 2

                ttc_val = float(ttc_los(veh_state, vru_p))
                ttc_min = min(ttc_min, ttc_val)
                if math.isfinite(ttc_val):
                    short = max(0.0, float(cfg.ttc_crit) - ttc_val)
                    if short > 0.0:
                        J_ttc += w_b * float(cfg.w_ttc) * (short ** 2)

            return ttc_min, (J_ttc + J_clear), collision

        # expand beam
        for k in range(N):
            tau = (k + 1) * dt
            new_beam: List[Tuple[float, VehState, float, float, List[float], List[float], float, int]] = []

            xr = float(x_ref[k + 1])
            yr = float(contract.y_center[k])

            for cost, veh, a_prev, d_prev, seq_a, seq_d, ttc_min_b, flips in beam:
                for a in a_cands:
                    for delta in delta_cands:
                        # stability caps for dry-run feasibility (Fix 5)
                        if stability_caps is not None:
                            jerk = (float(a) - float(a_prev)) / max(dt, 1e-6)
                            drate = (float(delta) - float(d_prev)) / max(dt, 1e-6)
                            if abs(jerk) > float(stability_caps.get("jerk", 1e9)):
                                continue
                            if abs(drate) > float(stability_caps.get("deltarate", 1e9)):
                                continue

                        veh1 = self._step_vehicle_dt(veh, a=float(a), delta=float(delta), dt_override=dt)

                        # hard road bounds
                        if not (float(cfg.y_road_min) - 1e-6 <= float(veh1.y) <= float(cfg.y_road_max) + 1e-6):
                            continue

                        # hard corridor (Fix 4)
                        ylo = float(contract.y_lo[k])
                        yhi = float(contract.y_hi[k])

                        # controlled corridor widen only if emergency (Fix 4)
                        ttc_min_step, J_safety, collision = compute_ttc_and_clearance_cost(veh1, tau)
                        if collision:
                            continue
                        emergency = bool(ttc_min_step < float(cfg.ttc_crit))
                        if emergency:
                            widen = float(cfg.corridor_widen_emergency)
                            ylo = max(float(cfg.y_road_min) + float(cfg.lane_margin), ylo - widen)
                            yhi = min(float(cfg.y_road_max) - float(cfg.lane_margin), yhi + widen)

                        if not (ylo - 1e-6 <= float(veh1.y) <= yhi + 1e-6):
                            continue

                        # YIELD do-not-cross stopline
                        if contract.x_max is not None and float(veh1.x) > float(contract.x_max) + 1e-6:
                            continue

                        track_scale = float(cfg.track_scale_emergency if emergency else 1.0)

                        # stage cost
                        J = 0.0
                        J += track_scale * float(cfg.w_track_x) * (float(veh1.x) - xr) ** 2
                        J += track_scale * float(cfg.w_track_y) * (float(veh1.y) - yr) ** 2
                        J += float(cfg.w_speed) * (float(veh1.v) - float(contract.v_ref[k])) ** 2
                        J += float(cfg.w_acc) * (float(a) ** 2)
                        J += float(cfg.w_delta) * (float(delta) ** 2)

                        jerk = (float(a) - float(a_prev)) / max(dt, 1e-6)
                        drate = (float(delta) - float(d_prev)) / max(dt, 1e-6)
                        J += float(cfg.w_jerk) * (jerk ** 2)
                        J += float(cfg.w_deltarate) * (drate ** 2)

                        J += float(J_safety)

                        if k == N - 1:
                            J += float(cfg.w_terminal_y) * (float(veh1.y) - float(contract.y_center[-1])) ** 2
                            J += float(cfg.w_terminal_yaw) * (float(veh1.yaw) ** 2)

                        # sign flips (for feasibility debugging)
                        flips_new = int(flips)
                        if seq_d:
                            prev = float(seq_d[-1])
                            if (prev > 1e-4 and float(delta) < -1e-4) or (prev < -1e-4 and float(delta) > 1e-4):
                                flips_new += 1
                                if stability_caps is not None:
                                    max_flips = int(stability_caps.get("sign_flips", 1_000_000))
                                    if flips_new > max_flips:
                                        continue

                        ttc_min_new = min(float(ttc_min_b), float(ttc_min_step))
                        new_beam.append((cost + J, veh1, float(a), float(delta), seq_a + [float(a)], seq_d + [float(delta)], ttc_min_new, flips_new))

            new_beam.sort(key=lambda z: z[0])
            beam = new_beam[:beam_width]

            if debug_costs:
                beam_log.append(
                    [
                        {
                            "k": k,
                            "tau": tau,
                            "cost": float(node[0]),
                            "a0": float(node[4][0]) if node[4] else None,
                            "d0_deg": math.degrees(float(node[5][0])) if node[5] else None,
                            "ttc_min_b": float(node[6]),
                            "flips": int(node[7]),
                        }
                        for node in beam[: min(5, len(beam))]
                    ]
                )

            if not beam:
                break

        corridor_dbg = {
            "dt": dt,
            "y_lo": list(contract.y_lo),
            "y_hi": list(contract.y_hi),
            "y_center": list(contract.y_center),
            "x_ref": x_ref[1:],
            "v_ref": list(contract.v_ref),
            "x_max": contract.x_max,
            "mode": contract.mode,
        }

        if not beam:
            a_brake = min(a_cands) if a_cands else -3.0
            return {
                "a_cmd": float(a_brake),
                "delta_cmd": 0.0,
                "feasible": False,
                "ttc_min_b": float("inf"),
                "ttc_trigger": False,
                "corridor": corridor_dbg,
                **({"beam_log": beam_log} if debug_costs else {}),
            }

        # prefer TTC-safe node if exists
        safe_nodes = [node for node in beam if float(node[6]) >= float(cfg.ttc_crit)]
        chosen = min(safe_nodes, key=lambda z: z[0]) if safe_nodes else min(beam, key=lambda z: z[0])

        _, _, _, _, seq_a, seq_d, ttc_min_b, _ = chosen
        a_cmd = float(seq_a[0]) if seq_a else 0.0
        delta_cmd = float(seq_d[0]) if seq_d else float(delta0)

        out = {
            "a_cmd": float(a_cmd),
            "delta_cmd": float(delta_cmd),
            "feasible": True,
            "ttc_min_b": float(ttc_min_b),
            "ttc_trigger": bool(float(ttc_min_b) < float(cfg.ttc_crit)),
            "corridor": corridor_dbg,
        }
        if debug_costs:
            out["beam_log"] = beam_log
        return out

    def _tactical_feasibility_certificate(
        self,
        *,
        veh0: VehState,
        vru0: VRUState,
        contract: MPC2DContract,
        belief: Dict[str, float],
        delta0: float,
        vru_pred_by_type: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = self.cfg
        mode = contract.mode

        if mode == "DETOUR_LEFT" or mode == "DETOUR_RIGHT":
            N = int(cfg.dry_steps_detour)
        elif mode == "YIELD":
            N = int(cfg.dry_steps_yield)
        else:
            N = int(cfg.dry_steps_keep)

        dry = MPC2DContract(
            mode=contract.mode,
            dt=float(contract.dt),
            N=N,
            y_center=contract.y_center[:N],
            y_lo=contract.y_lo[:N],
            y_hi=contract.y_hi[:N],
            v_ref=contract.v_ref[:N],
            x_max=contract.x_max,
            commit_until_t=contract.commit_until_t,
            terminal=contract.terminal,
        )

        # Analytical stopping-distance pre-check (Fix F)
        if mode == "YIELD" and contract.x_max is not None:
            _a_min_f = min(float(a.a) for a in self.AV) if self.AV else -5.0
            _v_f = float(veh0.v)
            _x_stop_f = float(veh0.x) + (_v_f * _v_f) / (2.0 * max(abs(_a_min_f), 1e-3))
            if float(veh0.x) < float(contract.x_max) and _x_stop_f > float(contract.x_max):
                return {"feasible": False, "reason": "stopping_distance_exceeds_x_max", "ttc_min_b": None}

        caps = {"jerk": float(cfg.dry_max_jerk), "deltarate": float(cfg.dry_max_deltarate), "sign_flips": int(cfg.dry_max_sign_flips)}

        _v0_cert = float(veh0.v)
        _a_hot_cert = min(float(a.a) for a in self.AV) if self.AV else -5.0
        out = self._tactical_beam_search_contract(
            veh0=veh0,
            vru0=vru0,
            contract=dry,
            belief=belief,
            delta0=float(delta0),
            vru_pred_by_type=vru_pred_by_type,
            debug_costs=False,
            beam_width_override=int(cfg.dry_beam_width),
            delta_candidates_deg_override=cfg.dry_delta_candidates_deg,
            stability_caps=caps,
            a_prev_override=_a_hot_cert if (contract.mode == "YIELD" and _v0_cert > float(cfg.v_yield_approach)) else None,
        )

        if not bool(out.get("feasible", False)):
            return {"feasible": False, "reason": "no_feasible_beam", "ttc_min_b": out.get("ttc_min_b")}

        # sign flip cap (stability)
        # approximate using chosen plan from beam_log is not available; instead re-run with debug_costs and inspect top node flips.
        # cheap: accept; in most cases jerk/drate caps prevent oscillations.
        return {"feasible": True, "reason": None, "ttc_min_b": out.get("ttc_min_b")}

    # -----------------------------
    # Undetected lane return
    # -----------------------------
    def _lane_return_delta(self, *, veh0: VehState, y_ref: float) -> float:
        cfg = self.cfg
        Ld = max(2.0, float(cfg.return_lookahead))
        tx = float(veh0.x) + Ld
        ty = float(_clamp(y_ref, cfg.y_road_min, cfg.y_road_max))

        alpha = math.atan2(ty - float(veh0.y), tx - float(veh0.x)) - float(veh0.yaw)
        alpha = _wrap_pi(alpha)

        WB = float(veh0.params.WB)
        delta = math.atan2(2.0 * WB * math.sin(alpha), Ld)

        dmax = math.radians(float(cfg.delta_max_deg))
        return float(_clamp(delta, -dmax, dmax))

    # -----------------------------
    # Belief + cruise
    # -----------------------------
    def _belief_probs(self, belief: Any) -> Dict[str, float]:
        if belief is None:
            raise ValueError("belief is mandatory")
        if hasattr(belief, "probs"):
            b = dict(belief.probs())
        else:
            b = {
                "Aggressive": float(getattr(belief, "p_aggr", belief.get("Aggressive", 0.5))),
                "Conservative": float(belief.get("Conservative", 0.5)),
            }
        s = float(b.get("Aggressive", 0.0) + b.get("Conservative", 0.0))
        if s <= 1e-9:
            return {"Aggressive": 0.5, "Conservative": 0.5}
        return {
            "Aggressive": float(b.get("Aggressive", 0.0)) / s,
            "Conservative": float(b.get("Conservative", 0.0)) / s,
        }

    def _cruise_accel(self, *, veh0: VehState, ref_speed: float) -> float:
        cfg = self.cfg
        a = float(cfg.cruise_kp) * (float(ref_speed) - float(veh0.v))
        return float(_clamp(a, float(cfg.cruise_a_min), float(cfg.cruise_a_max)))

    # -----------------------------
    # Utilities + logging
    # -----------------------------
    def _step_vehicle_dt(self, s: VehState, *, a: float, delta: float, dt_override: float) -> VehState:
        ns = copy_veh(s)
        ns.dt = float(dt_override)
        return step_vehicle(ns, VehicleAction(float(a)), float(delta))

    def _make_vru_interp(self, traj: List[VRUState], times: List[float]) -> Callable[[float], VRUState]:
        if not traj:
            raise ValueError("empty vru trajectory for interpolation")
        if len(traj) != len(times):
            dt = float(traj[0].dt)
            times = [i * dt for i in range(len(traj))]

        def interp(tau: float) -> VRUState:
            tau = float(tau)
            if tau <= float(times[0]):
                return traj[0]
            if tau >= float(times[-1]):
                return traj[-1]
            # linear search is fine at small horizon
            for i in range(1, len(times)):
                if tau <= float(times[i]):
                    t0, t1 = float(times[i - 1]), float(times[i])
                    s = (tau - t0) / max(1e-9, (t1 - t0))
                    a0, a1 = traj[i - 1], traj[i]
                    out = copy_vru(a0)
                    out.x = float(a0.x) + s * (float(a1.x) - float(a0.x))
                    out.y = float(a0.y) + s * (float(a1.y) - float(a0.y))
                    # out.v = float(a0.v) + s * (float(a1.v) - float(a0.v))
                    # out.yaw = float(a0.yaw) + s * (float(a1.yaw) - float(a0.yaw))
                    return out
            return traj[-1]

        return interp

    def _compute_a0_by_type(self, *, veh0: VehState, vru0: VRUState, ref_speed: float, a_cmd: float, delta_cmd: float) -> Dict[str, Any]:
        ego_plan = EgoPlan(a_seq=[VehicleAction(float(a_cmd))], delta_seq=[float(delta_cmd)])
        out: Dict[str, Any] = {}
        for rt in ("Aggressive", "Conservative"):
            ws = rider_weights(rt)
            roll = self.oracle_1.rollout_best_response(
                veh0=veh0,
                vru0=vru0,
                ego_plan=ego_plan,
                ref_speed=float(ref_speed),
                weights_vehicle=self.wv,
                weights_vru=ws,
            )
            out[rt] = roll.a0
        return out

    def _log_event(self, ev: Dict[str, Any]) -> None:
        self._events.append(dict(ev))
        if self._events_path:
            try:
                with open(self._events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev) + "\n")
            except Exception:
                pass

    def _append_tick(self, t: float, veh0: VehState, vru0: VRUState, dbg: Dict[str, Any], override: Optional[Dict[str, Any]]) -> None:
        rec = {
            "t": float(t),
            "mode": dbg.get("contract", {}).get("mode") if isinstance(dbg.get("contract"), dict) else None,
            "override": override,
            "ttc_min_b": dbg.get("ttc_min_b"),
            "ttc_trigger": dbg.get("ttc_trigger"),
            "mismatch": dbg.get("mismatch"),
            "veh": {"x": float(veh0.x), "y": float(veh0.y), "v": float(veh0.v), "yaw": float(veh0.yaw)},
            "vru": {
                "x": float(vru0.x),
                "y": float(vru0.y),
                "vx": float(getattr(vru0, "vx", 0.0)),
                "vy": float(getattr(vru0, "vy", 0.0)),
                # Keep key name "v" for backward-compatible logging: interpret as speed magnitude.
                "v": float(math.hypot(getattr(vru0, "vx", 0.0), getattr(vru0, "vy", 0.0))),
                # VRUState does not provide yaw; derive it from velocity direction when moving.
                "yaw": float(
                    math.atan2(getattr(vru0, "vy", 0.0), getattr(vru0, "vx", 0.0))
                    if math.hypot(getattr(vru0, "vx", 0.0), getattr(vru0, "vy", 0.0)) > 1e-6
                    else 0.0
                ),
            },
        }
        self._tick_hist.append(rec)
        if self._ticks_path:
            try:
                with open(self._ticks_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

    def _dump_debug_plots(self, *args, **kwargs) -> None:
        """No-op placeholder to avoid AttributeError. Plotting should be done by the renderer."""
        return