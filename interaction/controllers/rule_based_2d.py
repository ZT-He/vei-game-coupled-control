"""interaction/controllers/rule_based_2d.py

2D rule-based + prediction (longitudinal + lateral).

Implements the FSM you described:
  KEEP_SET_SPEED -> DETECT_VRU -> PREDICT_VRU -> SELECT_MIN_RISK_PATH -> CHECK_SAFETY -> (KEEP_SET_SPEED or BRAKE)

Key idea (occupancy-map / time-to-occupancy inspired)
----------------------------------------------------
1) When VRU is detected (FOV), predict VRU motion over a horizon H (fixed follower oracle).
2) Build a lightweight time-indexed "occupancy" representation from the predicted VRU trajectory.
3) Generate a small set of candidate detour reference paths (bounded within lane y ∈ [y_min, y_max]).
4) Score each candidate by an aggregated collision-risk metric over the horizon.
5) Track the selected reference path with Pure Pursuit, while using PID for speed tracking.
6) If risk is too high / no feasible detour exists, smoothly brake (PID + optional jerk limit).

Outputs
-------
- a_cmd: longitudinal acceleration command (m/s^2)
- delta_cmd: steering command (rad)

Both the ego and VRU predicted trajectories remain plottable in your renderer
because the sim loop still logs x_traj/y_traj, and this controller always
returns valid commands each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from interaction.controllers.base import EgoController, EgoControlOutput
from interaction.game.dynamics import VehicleAction, VRUAction
from interaction.game.payoff import PayoffWeights, vehicle_radius
from interaction.game.game_mpc import rider_weights
from interaction.oracle import InteractionOracle, EgoPlan


# =============================================================================
# Small helpers
# =============================================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _smoothstep01(s: float) -> float:
    """C1 smooth step from 0->1 for s in [0,1]."""
    s = _clamp(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass
class PID:
    """Minimal PID controller."""
    kp: float
    ki: float
    kd: float
    integrator_limit: float = 5.0

    _e_int: float = 0.0
    _e_prev: Optional[float] = None

    def reset(self) -> None:
        self._e_int = 0.0
        self._e_prev = None

    def step(self, *, e: float, dt: float) -> float:
        dt = max(float(dt), 1e-6)

        self._e_int += float(e) * dt
        self._e_int = _clamp(self._e_int, -self.integrator_limit, +self.integrator_limit)

        if self._e_prev is None:
            de = 0.0
        else:
            de = (float(e) - float(self._e_prev)) / dt
        self._e_prev = float(e)

        return float(self.kp * e + self.ki * self._e_int + self.kd * de)


# =============================================================================
# FSM definition
# =============================================================================

class Rule2DState(str, Enum):
    KEEP_SET_SPEED = "KEEP_SET_SPEED"
    DETECT_VRU = "DETECT_VRU"
    PREDICT_VRU = "PREDICT_VRU"
    SELECT_MIN_RISK_PATH = "SELECT_MIN_RISK_PATH"
    CHECK_SAFETY = "CHECK_SAFETY"
    BRAKE = "BRAKE"


# =============================================================================
# Config
# =============================================================================

@dataclass
class Rule2DConfig:
    # Horizon used for VRU prediction + risk evaluation
    horizon_steps: int = 20

    # Lane boundary constraint (your requirement)
    y_min: float = -7.0
    y_max: float = -1.0

    # Candidate absolute y targets for detour reference paths (edit to match lane centers)
    candidate_y_targets: Tuple[float, ...] = (-2.0, -3.0, -4.0, -5.0)

    # Smooth lateral transition duration (steps)
    lateral_transition_steps: int = 8

    # Pure pursuit
    lookahead_base: float = 6.0
    lookahead_k: float = 0.35
    delta_max_deg: float = 25.0

    # Risk model (aggregate proximity risk over horizon)
    risk_sigma: float = 2.0
    risk_threshold: float = 1.00
    time_weight: float = 0.10

    # TTC thresholds for braking logic (independent from risk_threshold)
    brake_if_pred_ttc_below: float = 4.0   # [R2D-3] raised from 3.0; v_target denominator now matches Rule1D
    exit_brake_if_ttc_above: float = 4.5

    # [R2D-2] After passing VRU conflict zone by this margin (m), force exit BRAKE.
    conflict_clear_margin: float = 1.0

    # [R2D-1] Nominal acceleration for BRAKE exit re-prediction (m/s^2).
    # 0.0 = constant-speed oracle; mirrors Rule1D Fix 2.
    brake_exit_a_nom: float = 0.0

    # Longitudinal PID
    cruise_pid: PID = field(default_factory=lambda: PID(kp=1.0, ki=0.15, kd=0.05, integrator_limit=6.0))
    brake_pid: PID = field(default_factory=lambda: PID(kp=1.4, ki=0.10, kd=0.08, integrator_limit=6.0))
    a_brake_max: float = 5.0

    # Optional jerk limit (m/s^3). Set None to disable.
    jerk_limit: Optional[float] = 15.0


# =============================================================================
# Controller
# =============================================================================

class RuleBased2DController(EgoController):
    name = "rule2d"

    def __init__(
        self,
        *,
        oracle: InteractionOracle,
        weights_vehicle: PayoffWeights,
        config: Rule2DConfig | None = None,
        verbose: bool = False,
    ):
        self.oracle = oracle
        self.wv = weights_vehicle
        self.cfg = config or Rule2DConfig(horizon_steps=oracle.H)
        self._verbose = bool(verbose)

        self.state: Rule2DState = Rule2DState.KEEP_SET_SPEED

        # Stored reference polyline (x,y) for pure pursuit
        self._ref_x: np.ndarray = np.array([], dtype=float)
        self._ref_y: np.ndarray = np.array([], dtype=float)

        # Cached results from SELECT_MIN_RISK_PATH used in CHECK_SAFETY
        self._risk_best: float = float("inf")
        self._feasible_best: bool = False
        self._min_ttc_pred: float = float("inf")
        self._type_hat: str = "Conservative"
        self._y_best: Optional[float] = None

        # Occupancy list built in PREDICT_VRU: [(x,y,k), ...]
        self._tto_occ: List[Tuple[float, float, float]] = []

        # For jerk limiting
        self._a_last: float = 0.0

        # Detection transition tracking (I5): reset cruise PID derivative on True→False
        # to prevent a derivative spike on the first cruise step after a prediction phase.
        self._prev_detected: bool = False

    # ---------------- belief helpers ----------------

    def _belief_probs(self, belief: Any) -> Dict[str, float]:
        if belief is None:
            raise ValueError("belief is mandatory (Step 3)")
        if hasattr(belief, "probs"):
            b = dict(belief.probs())
        else:
            b = {"Aggressive": float(belief.get("Aggressive", 0.5)),
                 "Conservative": float(belief.get("Conservative", 0.5))}
        s = float(b.get("Aggressive", 0.0) + b.get("Conservative", 0.0))
        if s <= 1e-9:
            return {"Aggressive": 0.5, "Conservative": 0.5}
        return {"Aggressive": b.get("Aggressive", 0.0) / s,
                "Conservative": b.get("Conservative", 0.0) / s}

    def _map_type(self, belief: Any) -> str:
        b = self._belief_probs(belief)
        return "Aggressive" if float(b.get("Aggressive", 0.5)) >= 0.5 else "Conservative"

    # ---------------- path generation ----------------

    def _forward_x(self, *, veh0, a_nom: float) -> np.ndarray:
        """Forward progress x(t) under constant accel a_nom (simple kinematics)."""
        H = int(self.cfg.horizon_steps)
        dt = float(veh0.dt)
        v0 = float(veh0.v)
        x0 = float(veh0.x)
        t = np.arange(H + 1, dtype=float) * dt
        return x0 + v0 * t + 0.5 * float(a_nom) * (t ** 2)

    def _nominal_path(self, *, veh0, a_nom: float) -> Tuple[np.ndarray, np.ndarray]:
        x = self._forward_x(veh0=veh0, a_nom=a_nom)
        y0 = _clamp(float(veh0.y), self.cfg.y_min, self.cfg.y_max)
        y = np.full_like(x, y0)
        return x, y

    def _detour_path(self, *, veh0, a_nom: float, y_target: float) -> Tuple[np.ndarray, np.ndarray]:
        x = self._forward_x(veh0=veh0, a_nom=a_nom)
        H = int(self.cfg.horizon_steps)
        y0 = float(veh0.y)
        y_target = _clamp(float(y_target), self.cfg.y_min, self.cfg.y_max)

        n_tr = int(max(1, self.cfg.lateral_transition_steps))
        y = np.empty(H + 1, dtype=float)
        for k in range(H + 1):
            if k <= n_tr:
                s = float(k) / float(n_tr)
                blend = _smoothstep01(s)
                y[k] = y0 + (y_target - y0) * blend
            else:
                y[k] = y_target

        y = np.clip(y, self.cfg.y_min, self.cfg.y_max)
        return x, y

    # ---------------- pure pursuit ----------------

    def _pure_pursuit_delta_raw(self, *, veh0, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
        """Return *unclamped* pure pursuit steering delta."""
        if ref_x.size < 2:
            return 0.0

        Ld = float(self.cfg.lookahead_base + self.cfg.lookahead_k * max(0.0, float(veh0.v)))
        Ld = max(2.0, Ld)

        x = float(veh0.x)
        y = float(veh0.y)
        yaw = float(veh0.yaw)

        dx = ref_x - x
        dy = ref_y - y
        dist = np.hypot(dx, dy)

        # Exclude path points that are behind the vehicle to prevent reverse steering
        # when the vehicle has passed reference points. A point is "ahead" when its
        # displacement has a positive dot product with the vehicle heading vector.
        heading_x = math.cos(yaw)
        heading_y = math.sin(yaw)
        forward = dx * heading_x + dy * heading_y  # >0 means ahead of vehicle

        ahead = forward > 0.0
        ahead_and_far = ahead & (dist >= Ld)

        if np.any(ahead_and_far):
            # First ahead point at least Ld away.
            idx = int(np.argmax(ahead_and_far))
        elif np.any(ahead):
            # No ahead point is >= Ld; use the furthest ahead point.
            idx = int(np.argmax(np.where(ahead, dist, -np.inf)))
        else:
            # All points behind (e.g. vehicle overshot path end); use last point.
            idx = int(ref_x.size - 1)

        xt = float(ref_x[idx])
        yt = float(ref_y[idx])

        alpha = _wrap_pi(math.atan2(yt - y, xt - x) - yaw)

        L = float(getattr(veh0.params, "WB", 2.5))
        return float(math.atan2(2.0 * L * math.sin(alpha), max(Ld, 1e-6)))

    def _pure_pursuit_delta(self, *, veh0, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
        """Return clamped steering delta."""
        delta_raw = self._pure_pursuit_delta_raw(veh0=veh0, ref_x=ref_x, ref_y=ref_y)
        delta_max = math.radians(float(self.cfg.delta_max_deg))
        return _clamp(delta_raw, -delta_max, +delta_max)

    # ---------------- risk model ----------------

    def _build_tto_occ(self, *, vru_traj) -> List[Tuple[float, float, float]]:
        return [(float(s.x), float(s.y), float(k)) for k, s in enumerate(vru_traj)]

    def _risk_of_path(self, *, ref_x: np.ndarray, ref_y: np.ndarray, occ: List[Tuple[float, float, float]], veh0, vru0) -> float:
        if ref_x.size == 0 or not occ:
            return 0.0

        sigma = float(max(1e-3, self.cfg.risk_sigma))
        inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

        # Hard collision threshold for big penalty
        col_r = vehicle_radius(veh0) + float(getattr(vru0.params, "R", 0.5)) + 0.2

        n = int(min(ref_x.size, len(occ)))
        risk = 0.0
        for k in range(n):
            ex, ey = float(ref_x[k]), float(ref_y[k])
            vx, vy, tk = occ[k]
            d = math.hypot(vx - ex, vy - ey)

            if d <= col_r:
                risk += 1e3
                continue

            w_time = 1.0 / (1.0 + float(self.cfg.time_weight) * float(tk))
            risk += math.exp(-(d * d) * inv_2sig2) * w_time

        return float(risk)

    # ---------------- longitudinal commands ----------------

    def _cruise_accel(self, *, veh0, ref_speed: float) -> float:
        e = float(ref_speed) - float(veh0.v)
        a = self.cfg.cruise_pid.step(e=e, dt=float(veh0.dt))
        return _clamp(a, float(veh0.params.a_min), float(veh0.params.a_max))

    def _brake_accel(self, *, veh0, target_speed: float) -> float:
        e = float(target_speed) - float(veh0.v)
        a = self.cfg.brake_pid.step(e=e, dt=float(veh0.dt))
        return _clamp(a, -abs(float(self.cfg.a_brake_max)), 0.0)

    def _apply_jerk_limit(self, *, a_cmd: float, dt: float) -> float:
        if self.cfg.jerk_limit is None:
            self._a_last = float(a_cmd)
            return float(a_cmd)

        jmax = float(self.cfg.jerk_limit)
        da_max = jmax * max(float(dt), 1e-6)
        a_limited = _clamp(float(a_cmd), self._a_last - da_max, self._a_last + da_max)
        self._a_last = float(a_limited)
        return float(a_limited)

    # ---------------- oracle helpers ----------------

    def _oracle_rollout(self, *, veh0, vru0, ego_plan: EgoPlan, ref_speed: float, rider_type: str):
        ws = rider_weights(rider_type)
        return self.oracle.rollout_best_response(
            veh0=veh0,
            vru0=vru0,
            ego_plan=ego_plan,
            ref_speed=ref_speed,
            weights_vehicle=self.wv,
            weights_vru=ws,
        )

    def _oracle_min_ttc(self, roll) -> float:
        if not roll.trace:
            return 1e9
        return float(min(float(row.get("ttc", 1e9)) for row in roll.trace))

    def _oracle_a0_by_type(self, *, veh0, vru0, ego_plan: EgoPlan, ref_speed: float) -> Dict[str, VRUAction]:
        out: Dict[str, VRUAction] = {}
        for rt in ("Aggressive", "Conservative"):
            roll = self._oracle_rollout(veh0=veh0, vru0=vru0, ego_plan=ego_plan, ref_speed=ref_speed, rider_type=rt)
            out[rt] = roll.a0
        return out

    # =============================================================================
    # Main plan
    # =============================================================================

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

        # We'll update these within the FSM and return at the end.
        a_cmd = 0.0
        delta_cmd = float(delta0)

        # I5: on detected True→False transition, reset derivative term so the first
        # cruise step after a prediction/brake phase does not spike from a stale error.
        if self._prev_detected and not bool(detected):
            self.cfg.cruise_pid._e_prev = None
        self._prev_detected = bool(detected)

        # Nominal accel estimate for candidate path generation
        a_nom = self._cruise_accel(veh0=veh0, ref_speed=ref_speed)

        # Init ref path if needed
        if self._ref_x.size == 0:
            self._ref_x, self._ref_y = self._nominal_path(veh0=veh0, a_nom=a_nom)

        # Multi-transition per tick (prevents 1-tick lag), capped for safety.
        for _ in range(8):
            if self.state == Rule2DState.KEEP_SET_SPEED:
                a_cmd = a_nom
                delta_cmd = self._pure_pursuit_delta(veh0=veh0, ref_x=self._ref_x, ref_y=self._ref_y)
                self.state = Rule2DState.DETECT_VRU
                continue

            if self.state == Rule2DState.DETECT_VRU:
                if not bool(detected):
                    self._ref_x, self._ref_y = self._nominal_path(veh0=veh0, a_nom=a_nom)
                    self._risk_best = float("inf")
                    self._feasible_best = False
                    self._min_ttc_pred = float("inf")
                    self._tto_occ = []
                    self._y_best = None
                    self.state = Rule2DState.KEEP_SET_SPEED
                    break  # a_cmd/delta_cmd already set by KEEP_SET_SPEED above
                self.state = Rule2DState.PREDICT_VRU
                continue

            if self.state == Rule2DState.PREDICT_VRU:
                # Predict VRU using the actual current steering (delta0) and the
                # previous step's actual command (_a_last) as the longitudinal nominal,
                # so the oracle sees what the vehicle is currently executing rather than
                # the cruise PID setpoint (consistent with Rule1D's PREDICT_VRU fix).
                self._type_hat = self._map_type(belief)
                ego_plan_nom = EgoPlan(a_seq=[VehicleAction(float(self._a_last))], delta_seq=[float(delta0)])

                roll = self._oracle_rollout(
                    veh0=veh0, vru0=vru0, ego_plan=ego_plan_nom, ref_speed=ref_speed, rider_type=self._type_hat
                )
                self._min_ttc_pred = self._oracle_min_ttc(roll)
                self._tto_occ = self._build_tto_occ(vru_traj=roll.vru_traj)

                self.state = Rule2DState.SELECT_MIN_RISK_PATH
                continue

            if self.state == Rule2DState.SELECT_MIN_RISK_PATH:
                # Generate candidates and select min-risk detour.
                cand_ys = [float(_clamp(float(veh0.y), self.cfg.y_min, self.cfg.y_max))]
                cand_ys += [float(y) for y in self.cfg.candidate_y_targets]
                cand_ys = sorted({float(_clamp(y, self.cfg.y_min, self.cfg.y_max)) for y in cand_ys})

                best_ref = None
                best_risk = float("inf")
                best_y = None
                best_feas = False

                delta_max = math.radians(float(self.cfg.delta_max_deg))

                for y_t in cand_ys:
                    rx, ry = self._detour_path(veh0=veh0, a_nom=a_nom, y_target=y_t)

                    # feasibility: check if *raw* steering required at current state is within bounds
                    d_raw = self._pure_pursuit_delta_raw(veh0=veh0, ref_x=rx, ref_y=ry)
                    feasible = (abs(float(d_raw)) <= delta_max)

                    risk = self._risk_of_path(ref_x=rx, ref_y=ry, occ=self._tto_occ, veh0=veh0, vru0=vru0)

                    if feasible and (risk < best_risk):
                        best_ref = (rx, ry)
                        best_risk = float(risk)
                        best_y = float(y_t)
                        best_feas = True

                if best_ref is None:
                    self._ref_x, self._ref_y = self._nominal_path(veh0=veh0, a_nom=a_nom)
                    self._risk_best = float("inf")
                    self._feasible_best = False
                    self._y_best = None
                else:
                    self._ref_x, self._ref_y = best_ref
                    self._risk_best = float(best_risk)
                    self._feasible_best = bool(best_feas)
                    self._y_best = best_y

                self.state = Rule2DState.CHECK_SAFETY
                continue

            if self.state == Rule2DState.CHECK_SAFETY:
                # Original risk-based entry: enter BRAKE if no feasible low-risk detour.
                # Risk (risk_best) is also used in SELECT_MIN_RISK_PATH to rank paths;
                # here it gates whether any selected path is safe enough to follow.
                # NOTE: BRAKE exit uses TTC (exit_brake_if_ttc_above) — asymmetric by design.
                # TODO: evaluate whether unifying entry/exit on TTC improves collision rate.
                safe_detour = bool(self._feasible_best) and (float(self._risk_best) < float(self.cfg.risk_threshold))

                if safe_detour:
                    a_cmd = a_nom
                    delta_cmd = self._pure_pursuit_delta(veh0=veh0, ref_x=self._ref_x, ref_y=self._ref_y)
                    self.state = Rule2DState.KEEP_SET_SPEED
                    break

                # If detour isn't safe, go braking.
                self.state = Rule2DState.BRAKE
                continue

            if self.state == Rule2DState.BRAKE:
                # [R2D-2] Spatial exit: if vehicle front has passed VRU conflict zone,
                # the interaction is resolved — no need to keep braking.
                _C2R = float(getattr(getattr(veh0, "params", None), "C2R", 1.5))
                if float(veh0.x) - _C2R > float(vru0.x) + float(self.cfg.conflict_clear_margin):
                    self.cfg.brake_pid.reset()
                    self.state = Rule2DState.KEEP_SET_SPEED
                    break

                # Smooth brake based on predicted TTC.
                # Lower TTC -> lower target speed -> stronger braking PID.
                ttc_use = float(self._min_ttc_pred)
                scale = _clamp(ttc_use / max(self.cfg.brake_if_pred_ttc_below, 1e-3), 0.0, 1.0)
                v_target = float(ref_speed) * scale
                a_cmd = self._brake_accel(veh0=veh0, target_speed=v_target)

                # steer to nominal lane while braking (keeps behavior simple & stable).
                # Use a_cmd (actual brake accel) for consistent lookahead x-projection.
                nx, ny = self._nominal_path(veh0=veh0, a_nom=float(a_cmd))
                delta_cmd = self._pure_pursuit_delta(veh0=veh0, ref_x=nx, ref_y=ny)

                # [R2D-1 + R2D-4] Refresh TTC using constant-speed oracle, gated on detected.
                # Previously used a_cmd (negative) which caused a self-reinforcing loop:
                # braking ego → aggressive VRU doesn't yield → TTC stays low → stays in BRAKE.
                # Constant-speed oracle (brake_exit_a_nom=0.0) lets the VRU respond to a
                # non-braking ego, allowing TTC to recover when the conflict resolves.
                # When VRU is not detected, retain the last known prediction (safe: conservative).
                if bool(detected):
                    ego_plan_exit = EgoPlan(
                        a_seq=[VehicleAction(float(self.cfg.brake_exit_a_nom))],
                        delta_seq=[float(delta_cmd)],
                    )
                    roll_now = self._oracle_rollout(
                        veh0=veh0, vru0=vru0, ego_plan=ego_plan_exit,
                        ref_speed=ref_speed, rider_type=self._type_hat,
                    )
                    self._min_ttc_pred = self._oracle_min_ttc(roll_now)
                min_ttc_now = float(self._min_ttc_pred)

                if float(min_ttc_now) >= float(self.cfg.exit_brake_if_ttc_above):
                    self.state = Rule2DState.KEEP_SET_SPEED

                break

            raise RuntimeError(f"Unknown state {self.state}")

        # Prevent cruise PID integral wind-up during BRAKE: the PID is called
        # unconditionally above (to compute a_nom), but its integral should not
        # accumulate while the vehicle is intentionally held below ref_speed.
        # Resetting here means each post-BRAKE cruise step starts with integrator=0,
        # avoiding overshoot on BRAKE exit. The integral is intentionally NOT
        # preserved across BRAKE phases (unlike MPC1D which preserves integral but
        # resets derivative — cruise here needs a clean restart after forced braking).
        if self.state == Rule2DState.BRAKE:
            self.cfg.cruise_pid.reset()

        # Final smoothing
        a_cmd = self._apply_jerk_limit(a_cmd=a_cmd, dt=float(veh0.dt))

        # Provide a0_by_type so sim can roll the "true rider type".
        # Gated behind verbose to avoid 2 extra oracle calls every detected step.
        if self._verbose and bool(detected):
            ego_plan_now = EgoPlan(a_seq=[VehicleAction(float(a_cmd))], delta_seq=[float(delta_cmd)])
            a0_by_type = self._oracle_a0_by_type(veh0=veh0, vru0=vru0, ego_plan=ego_plan_now, ref_speed=ref_speed)
        else:
            a0_by_type = {"Aggressive": VRUAction("go"), "Conservative": VRUAction("go")}

        debug: Dict[str, Any] = {
            "policy": "rule2d_fsm_detour",
            "scenario": scenario,
            "state": str(self.state),
            "detected": bool(detected),
            "belief": b,
            "type_hat": str(self._type_hat),
            "a_nom": float(a_nom),
            "risk_best": float(self._risk_best),
            "feasible_best": bool(self._feasible_best),
            "y_best": None if self._y_best is None else float(self._y_best),
            "min_ttc_pred": float(self._min_ttc_pred),
            "a0_by_type": a0_by_type,
        }

        return EgoControlOutput(
            a_cmd=float(a_cmd),
            delta_cmd=float(delta_cmd),
            feasible=True,
            debug=debug,
        )
