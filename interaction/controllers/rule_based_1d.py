"""interaction/controllers/rule_based_1d.py

Rule-based (1D longitudinal) + prediction baseline.

This controller implements an explicit FSM (finite-state machine) that matches
your pseudo-code:

    state = KEEP_SET_SPEED

    loop each timestep:
        KEEP_SET_SPEED -> DETECT_VRU -> PREDICT_VRU -> CHECK_SAFETY
            -> (KEEP_SET_SPEED if safe else BRAKE)
        BRAKE -> (KEEP_SET_SPEED if safe_now else BRAKE)

Key points
----------
- 1D longitudinal only: output is (a_cmd, delta_cmd) where delta_cmd stays at
  delta0 (usually 0.0).
- Prediction uses the shared InteractionOracle to obtain a **VRU best response**
  to a nominal ego plan, under a single predicted rider type (MAP under belief).
  This keeps the follower policy fixed across experiments.
- Braking is smoothed using a small speed-PID toward a reduced “safe” target
  speed, plus an optional jerk limiter.

Notes
-----
- Belief is mandatory (Step 3). Rule1D uses belief only for prediction (select
  the MAP rider type), not for optimization.
- For paper reproducibility, all thresholds and gains are in Rule1DConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math

from interaction.oracle import InteractionOracle, EgoPlan
from interaction.controllers.base import EgoController, EgoControlOutput

from interaction.game.dynamics import VehicleAction
from interaction.game.payoff import PayoffWeights
from interaction.game.game_mpc import rider_weights


# ----------------------------- Config + helpers -----------------------------


@dataclass
class Rule1DConfig:
    """Configuration knobs for the rule-based FSM."""

    horizon_steps: int = 20

    # --- Safety thresholds (TTC-based) ---
    # If predicted minimum TTC < this threshold, enter BRAKE.
    brake_if_min_ttc_below: float = 4.0
    # Exit BRAKE only when predicted min TTC is comfortably above this.
    # Must be >= brake_if_min_ttc_below to form a proper hysteresis band.
    exit_brake_if_min_ttc_above: float = 4.5

    # --- Cruise PID (speed tracking) ---
    cruise_kp: float = 1.2
    cruise_ki: float = 0.05
    cruise_kd: float = 0.15

    # --- Brake PID (smooth deceleration) ---
    brake_kp: float = 1.8
    brake_ki: float = 0.08
    brake_kd: float = 0.20

    # Max decel used by rule-based brake (still clamped by vehicle a_min in dynamics).
    a_brake_max: float = 5.0  # m/s^2

    # Optional jerk limiter for comfort (set None to disable).
    jerk_limit: Optional[float] = 6.0  # m/s^3

    # When braking, scale v_set by this factor at min_ttc=0.
    # v_target = ref_speed * clamp(min_ttc / brake_if_min_ttc_below, floor..1)
    brake_speed_scale_floor: float = 0.0

    # [Fix 1] After passing VRU conflict zone by this margin (m), force exit BRAKE.
    conflict_clear_margin: float = 1.0

    # [Fix 2] Nominal acceleration used in BRAKE exit re-prediction (m/s^2).
    # 0.0 = constant-speed oracle; avoids the self-reinforcing braking loop where
    # a_nom=last_a_cmd (negative) causes the oracle to always see a braking vehicle,
    # keeping the aggressive VRU non-yielding and TTC perpetually low.
    brake_exit_a_nom: float = 0.0

    # [Fix 5] When min_ttc_pred falls below this (s), skip jerk smoothing for
    # deceleration commands. Safety > comfort at critical TTC.
    critical_ttc_jerk_bypass: float = 1.5

    def __post_init__(self) -> None:
        self.horizon_steps = int(self.horizon_steps)
        if self.horizon_steps <= 0:
            self.horizon_steps = 1


class _PID:
    """Tiny PID helper used for cruise/brake acceleration commands."""

    def __init__(self, kp: float, ki: float, kd: float, integ_limit: float = 10.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integ_limit = float(integ_limit)
        self.e_int = 0.0
        self.e_prev: Optional[float] = None

    def reset(self) -> None:
        self.e_int = 0.0
        self.e_prev = None

    def step(self, setpoint: float, meas: float, dt: float) -> float:
        dt = max(float(dt), 1e-6)
        e = float(setpoint) - float(meas)
        self.e_int = max(-self.integ_limit, min(self.integ_limit, self.e_int + e * dt))
        de = 0.0 if self.e_prev is None else (e - self.e_prev) / dt
        self.e_prev = e
        return self.kp * e + self.ki * self.e_int + self.kd * de


# ----------------------------- Controller implementation -----------------------------


class RuleBased1DController(EgoController):
    """1D rule-based controller with explicit FSM transitions."""

    name = "rule1d"

    # FSM state names (strings keep debug easy to read)
    KEEP_SET_SPEED = "KEEP_SET_SPEED"
    DETECT_VRU = "DETECT_VRU"
    PREDICT_VRU = "PREDICT_VRU"
    CHECK_SAFETY = "CHECK_SAFETY"
    BRAKE = "BRAKE"

    def __init__(
        self,
        *,
        oracle: InteractionOracle,
        weights_vehicle: PayoffWeights,
        config: Rule1DConfig | None = None,
        verbose: bool = False,
    ):
        self.oracle = oracle
        self.wv = weights_vehicle
        self.cfg = config or Rule1DConfig(horizon_steps=oracle.H)
        self._verbose = bool(verbose)

        # Internal controller memory
        self.state: str = self.KEEP_SET_SPEED
        self._cruise_pid = _PID(self.cfg.cruise_kp, self.cfg.cruise_ki, self.cfg.cruise_kd)
        self._brake_pid = _PID(self.cfg.brake_kp, self.cfg.brake_ki, self.cfg.brake_kd)
        self._last_a_cmd: float = 0.0

        # Cache the last prediction rollout (for debug/exit condition)
        self._last_pred_roll = None

        # Detection transition tracking (I5): reset cruise PID derivative on True→False
        # to prevent a derivative spike on the first cruise step after a prediction phase.
        self._prev_detected: bool = False

    # ----------------------------- Belief helpers -----------------------------

    def _belief_probs(self, belief: Any) -> Dict[str, float]:
        """Return normalized {Aggressive: p, Conservative: p}.

        Belief is mandatory (Step 3). We keep this helper compatible with the
        belief object you use in sim (supports `.probs()`), and also a dict-like
        fallback.
        """
        if belief is None:
            raise ValueError("belief is mandatory (Step 3)")

        if hasattr(belief, "probs"):
            b = dict(belief.probs())
        else:
            b = {
                "Aggressive": float(belief.get("Aggressive", 0.5)),
                "Conservative": float(belief.get("Conservative", 0.5)),
            }

        s = float(b.get("Aggressive", 0.0) + b.get("Conservative", 0.0))
        if s <= 1e-9:
            return {"Aggressive": 0.5, "Conservative": 0.5}

        return {
            "Aggressive": float(b.get("Aggressive", 0.0)) / s,
            "Conservative": float(b.get("Conservative", 0.0)) / s,
        }

    # ----------------------------- Prediction + TTC helpers -----------------------------

    def _nominal_ego_plan(self, *, a_nom: float, delta0: float) -> EgoPlan:
        """Return an ego plan with constant acceleration and constant steering.

        EgoPlan supports length-1 sequences; the oracle repeats the last entry
        automatically up to horizon H.
        """
        return EgoPlan(a_seq=[VehicleAction(float(a_nom))], delta_seq=[float(delta0)])

    @staticmethod
    def _min_ttc_from_roll(roll) -> float:
        """Extract minimum TTC from an oracle rollout trace."""
        if roll is None or not getattr(roll, "trace", None):
            return 1e9
        return float(min(float(row.get("ttc", 1e9)) for row in roll.trace))

    def _predict_vru_best_response(
        self,
        *,
        veh0,
        vru0,
        a_nom: float,
        delta0: float,
        ref_speed: float,
        type_hat: str,
    ):
        """Predict VRU best response under the MAP rider type."""
        ws = rider_weights(type_hat)
        plan = self._nominal_ego_plan(a_nom=float(a_nom), delta0=float(delta0))
        return self.oracle.rollout_best_response(
            veh0=veh0,
            vru0=vru0,
            ego_plan=plan,
            ref_speed=float(ref_speed),
            weights_vehicle=self.wv,
            weights_vru=ws,
        )

    # ----------------------------- Main planning entry -----------------------------

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
        """Return one-step ego control command (a_cmd, delta_cmd).

        The FSM transitions are executed in a small bounded loop so that
        DETECT->PREDICT->CHECK can happen within the same timestep (no multi-step
        reaction latency).
        """

        # 1) belief -> choose MAP type for prediction
        b = self._belief_probs(belief)
        p_aggr = float(b.get("Aggressive", 0.5))
        type_hat = "Aggressive" if p_aggr >= 0.5 else "Conservative"

        # 2) compute cruise command (speed PID)
        dt = float(getattr(veh0, "dt", 0.1))

        # I5: on detected True→False transition, reset derivative term so the first
        # cruise step after a prediction/brake phase does not spike from a stale error.
        if self._prev_detected and not bool(detected):
            self._cruise_pid.e_prev = None
        self._prev_detected = bool(detected)

        cruise_a = float(self._cruise_pid.step(setpoint=float(ref_speed), meas=float(veh0.v), dt=dt))

        # 3) helper: jerk-limit for comfort (optional)
        def jerk_limit(a_des: float) -> float:
            if self.cfg.jerk_limit is None:
                return float(a_des)
            j = float(self.cfg.jerk_limit)
            da_max = j * dt
            da = float(a_des) - float(self._last_a_cmd)
            da = max(-da_max, min(da_max, da))
            return float(self._last_a_cmd) + da

        # Default output is cruise (can be overwritten by BRAKE)
        a_cmd = float(cruise_a)
        min_ttc_pred = 1e9
        safe_pred = True

        # 4) run FSM transitions
        #    (bounded to avoid infinite loops if a bug sneaks in)
        for _ in range(6):
            if self.state == self.KEEP_SET_SPEED:
                # KEEP_SET_SPEED: speed tracking
                a_cmd = float(cruise_a)
                # next: detection check
                self.state = self.DETECT_VRU
                continue

            if self.state == self.DETECT_VRU:
                # DETECT_VRU: perception gate (FOV-based flag from sim)
                if not bool(detected):
                    # No observation -> cruise.
                    self._last_pred_roll = None
                    self._brake_pid.reset()
                    self.state = self.KEEP_SET_SPEED
                    break
                # Detected -> predict VRU reaction
                self.state = self.PREDICT_VRU
                continue

            if self.state == self.PREDICT_VRU:
                # PREDICT_VRU: oracle-based best response prediction.
                # Use _last_a_cmd (previous step's actual command) as the nominal
                # ego plan so the oracle sees what the vehicle is currently executing,
                # consistent with the re-rollout done inside BRAKE (line ~317).
                self._last_pred_roll = self._predict_vru_best_response(
                    veh0=veh0,
                    vru0=vru0,
                    a_nom=float(self._last_a_cmd),
                    delta0=float(delta0),
                    ref_speed=float(ref_speed),
                    type_hat=str(type_hat),
                )
                self.state = self.CHECK_SAFETY
                continue

            if self.state == self.CHECK_SAFETY:
                # CHECK_SAFETY: TTC-based guard on predicted interaction
                min_ttc_pred = float(self._min_ttc_from_roll(self._last_pred_roll))
                safe_pred = bool(min_ttc_pred >= float(self.cfg.brake_if_min_ttc_below))
                if safe_pred:
                    # Safe -> return to cruising
                    self._brake_pid.reset()
                    self.state = self.KEEP_SET_SPEED
                    break
                # Unsafe -> brake
                self.state = self.BRAKE
                continue

            if self.state == self.BRAKE:
                # BRAKE: smooth deceleration using PID to a reduced target speed.
                # smaller TTC => smaller target speed

                # [Fix 1] Spatial exit: if vehicle front has passed VRU conflict zone,
                # the interaction is resolved — no need to keep braking.
                _C2R = float(getattr(getattr(veh0, "params", None), "C2R", 1.5))
                if float(veh0.x) - _C2R > float(vru0.x) + float(self.cfg.conflict_clear_margin):
                    self._brake_pid.reset()
                    self.state = self.KEEP_SET_SPEED
                    break

                # Refresh prediction while braking so the exit condition is based
                # on up-to-date kinematics (important when both agents are moving).
                if bool(detected):
                    self._last_pred_roll = self._predict_vru_best_response(
                        veh0=veh0,
                        vru0=vru0,
                        a_nom=float(self.cfg.brake_exit_a_nom),  # [Fix 2] constant-speed oracle
                        delta0=float(delta0),
                        ref_speed=float(ref_speed),
                        type_hat=str(type_hat),
                    )

                min_ttc_pred = float(self._min_ttc_from_roll(self._last_pred_roll))
                denom = max(float(self.cfg.brake_if_min_ttc_below), 1e-3)
                scale = max(float(self.cfg.brake_speed_scale_floor), min(1.0, float(min_ttc_pred) / denom))
                v_target = float(ref_speed) * float(scale)

                a_brake = float(self._brake_pid.step(setpoint=float(v_target), meas=float(veh0.v), dt=dt))
                # clamp to negative accel only; cap magnitude
                a_cmd = max(-float(self.cfg.a_brake_max), min(0.0, float(a_brake)))

                # exit brake if the predicted TTC is safely above threshold
                if float(min_ttc_pred) >= float(self.cfg.exit_brake_if_min_ttc_above):
                    self.state = self.KEEP_SET_SPEED
                break

            # Unknown state -> safe reset
            self.state = self.KEEP_SET_SPEED
            break

        # 5) comfort: jerk limit final command
        # [Fix 5] Emergency bypass: when TTC is critical and we need harder braking,
        # skip jerk smoothing — the 0.8 s ramp delay consumes most of the stopping
        # distance available at short TTC.  Only bypasses deceleration (a_cmd < last).
        _crit_ttc = float(self.cfg.critical_ttc_jerk_bypass)
        if float(min_ttc_pred) < _crit_ttc and float(a_cmd) < float(self._last_a_cmd):
            pass  # use raw brake command; jerk comfort is secondary to collision avoidance
        else:
            a_cmd = float(jerk_limit(a_cmd))
        self._last_a_cmd = float(a_cmd)

        # I1: prevent cruise PID integral wind-up during BRAKE.
        # The cruise PID is called unconditionally above, so its integral accumulates
        # even when the vehicle is held below ref_speed by BRAKE. Resetting here
        # ensures the first post-BRAKE cruise step starts with a clean integrator.
        if self.state == self.BRAKE:
            self._cruise_pid.reset()

        # 6) Provide debug + true rider rollout hook (used by sim loop)
        dbg: Dict[str, Any] = {
            "policy": "rule1d_fsm",
            "fsm_state": str(self.state),
            "detected": bool(detected),
            "belief": b,
            "type_hat": str(type_hat),
            "scenario": str(scenario),
            "cruise_a": float(cruise_a),
            "a_cmd": float(a_cmd),
            "safe_pred": bool(safe_pred),
            "min_ttc_pred": float(min_ttc_pred),
            # True on the first step(s) after exiting BRAKE: state is KEEP_SET_SPEED
            # but a_cmd is still negative due to jerk limiting from a braking command.
            "transitioning_from_brake": bool(self._last_a_cmd < 0.0 and self.state == self.KEEP_SET_SPEED),
        }

        # For sim rollout under "true rider type": provide predicted a0 for both types.
        # This mirrors the MPC controller debug convention so your sim can do:
        #   out.debug["a0_by_type"][TRUE_RIDER]
        # Gated behind verbose to avoid doubling oracle call count on every detected step.
        a0_by_type: Dict[str, Any] = {}
        if self._verbose and bool(detected):
            for rt in ("Aggressive", "Conservative"):
                roll = self._predict_vru_best_response(
                    veh0=veh0,
                    vru0=vru0,
                    a_nom=float(a_cmd),
                    delta0=float(delta0),
                    ref_speed=float(ref_speed),
                    type_hat=str(rt),
                )
                a0_by_type[rt] = roll.a0
        dbg["a0_by_type"] = a0_by_type

        return EgoControlOutput(
            a_cmd=float(a_cmd),
            delta_cmd=float(delta0),  # 1D controller keeps steering fixed
            feasible=True,
            debug=dbg,
        )
