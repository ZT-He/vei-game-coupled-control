# quant_sim/quant_game_intersection.py
from __future__ import annotations

import os
import time
import math
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Iterable, Iterator

import numpy as np
import xlsxwriter as xwr

from motion.vehicle import BicycleModel, BicycleModel_params
from motion.vru import PointMassNewton, PointMass_params
from motion.social_force import sfm_force_to_target

from perception.vehicle import VehFOV
from perception.vru_awareness_msg import VRU_AWR

from evaluation.sustain_evaluation import SustainEvaluator
from evaluation.collision_check import check_collision

from interaction.game.belief import RiderTypeBelief
from interaction.game.state import VehState, VRUState
from interaction.game.dynamics import VehicleAction, VRUAction
from interaction.game.payoff import PayoffWeights

# Step-5: planner swap
from interaction.oracle import InteractionOracle
from interaction.controllers.factory import make_ego_controller

# =============================================================================
# Progress helpers
# =============================================================================
def _fmt_seconds(sec: float) -> str:
    sec = float(max(sec, 0.0))
    if sec < 60:
        return f"{sec:.1f}s"
    m = sec / 60.0
    if m < 60:
        return f"{m:.1f}min"
    h = m / 60.0
    return f"{h:.2f}h"


def _progress_line_controller(
    done_all: int,
    total_all: int,
    controller: str,
    ctrl_done: int,
    ctrl_total: int,
    avg_sec_per_ep: float,
    eta_sec: float,
) -> str:
    pct = 100.0 * done_all / max(total_all, 1)
    cpct = 100.0 * ctrl_done / max(ctrl_total, 1)
    return (
        f"[PROGRESS] all {done_all}/{total_all} ({pct:5.1f}%) | "
        f"ctrl {controller}: {ctrl_done}/{ctrl_total} ({cpct:5.1f}%) | "
        f"avg={avg_sec_per_ep:.2f}s/ep | ETA≈{_fmt_seconds(eta_sec)}"
    )


def _estimate_total_configs(
    v2x_profiles: List[str],
    rider_types: List[str],
    seeds: List[int],
    speeds: List[float],
    vru_cfgs: List[Tuple[float, float, float, float, float, float, float, float]],
) -> int:
    """Return total number of possible configs (cartesian product)."""
    return int(len(v2x_profiles) * len(rider_types) * len(seeds) * len(speeds) * len(vru_cfgs))


# =============================================================================
# Scenario helpers
# =============================================================================
class VRUPathManager:
    """
    Two-stage path:
      stage0: go to turn point
      stage1: go to destination

    yield: go to stop point with v_des=0
    """

    def __init__(self, stop_pt, turn_pt, dest_pt, arrive_eps: float = 1.2):
        self.stop_pt = tuple(stop_pt)
        self.turn_pt = tuple(turn_pt)
        self.dest_pt = tuple(dest_pt)
        self.arrive_eps = float(arrive_eps)
        self.stage = 0

    def _near(self, vru, pt) -> bool:
        return math.hypot(float(vru.x) - pt[0], float(vru.y) - pt[1]) <= self.arrive_eps

    def step_target(self, vru, vru_mode: str):
        # Switch to stage1 when passing the turn point
        if vru_mode == "go" and self.stage == 0 and self._near(vru, self.turn_pt):
            self.stage = 1

        # Yield: head to stop point with v_des=0
        if vru_mode == "yield":
            stop_y = float(self.stop_pt[1])

            # if already at/above stop_y, stop in place (do NOT chase stop_pt backwards)
            if float(vru.y) >= stop_y - 0.2:
                return (float(vru.x), float(vru.y)), 0.0

            # does not reach stop line
            return self.stop_pt, 0.0

        # Go: follow stage target with nominal desired speed
        if self.stage == 0:
            return self.turn_pt, 5.0
        return self.dest_pt, 5.0


def ttc_los(veh, vru) -> float:
    """Line-of-sight TTC. If not closing => inf."""
    rx = float(vru.x) - float(veh.x)
    ry = float(vru.y) - float(veh.y)
    dist = math.hypot(rx, ry)
    if dist < 1e-6:
        return 0.0

    # NOTE: your BicycleModel appears to maintain a yaw-like heading.
    vvx = float(veh.v) * math.cos(float(veh.yaw))
    vvy = float(veh.v) * math.sin(float(veh.yaw))

    rvx = float(vru.v_x) - vvx
    rvy = float(vru.v_y) - vvy

    closing = -(rx * rvx + ry * rvy) / dist
    if closing <= 1e-6:
        return float("inf")
    return dist / closing

# =============================================================================
# Episode config + generators
# =============================================================================
@dataclass(frozen=True)
class EpisodeConfig:
    v2x_profile: str
    rider_type: str
    veh_speed: float
    seed: int

    vru_x0: float
    vru_y0: float
    stop_x: float
    stop_y: float
    turn_x: float
    turn_y: float
    dest_x: float
    dest_y: float


def build_vru_configs_default() -> List[Tuple[float, float, float, float, float, float, float, float]]:
    """
    Default VRU combinations (edit here later if needed).
    Returns list of tuples:
      (vru_x0, vru_y0, stop_x, stop_y, turn_x, turn_y, dest_x, dest_y)
    """
    x0_list = [4.0]
    y0_list = [-30, -25.0, -20.0, -15.0, -10.0]
    dest_list = [(-18.0, 20.0)]

    stop_y = -14.0
    turn_y = 0.0

    cfgs: List[Tuple[float, float, float, float, float, float, float, float]] = []
    for x0 in x0_list:
        for y0 in y0_list:
            for dx, dy in dest_list:
                cfgs.append((x0, y0, x0, stop_y, x0, turn_y, dx, dy))
    return cfgs


def iter_episode_configs(
    v2x: str,
    rider: str,
    speeds: List[float],
    seeds: List[int],
    vru_cfgs: List[Tuple[float, float, float, float, float, float, float, float]],
) -> Iterator[EpisodeConfig]:
    """Stream episode configs (avoid building a huge list in memory)."""
    for (x0, y0, sx, sy, tx, ty, dx, dy) in vru_cfgs:
        for spd in speeds:
            for sd in seeds:
                yield EpisodeConfig(
                    v2x_profile=v2x,
                    rider_type=rider,
                    veh_speed=float(spd),
                    seed=int(sd),
                    vru_x0=float(x0),
                    vru_y0=float(y0),
                    stop_x=float(sx),
                    stop_y=float(sy),
                    turn_x=float(tx),
                    turn_y=float(ty),
                    dest_x=float(dx),
                    dest_y=float(dy),
                )


def reservoir_sample(stream: Iterable[EpisodeConfig], k: int, rng: np.random.Generator) -> List[EpisodeConfig]:
    """Reservoir sampling: sample k items uniformly from a stream."""
    buf: List[EpisodeConfig] = []
    for i, item in enumerate(stream):
        if i < k:
            buf.append(item)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                buf[j] = item
    return buf


# =============================================================================
# Speed sampling (explicit: 10 or 100 points)
# =============================================================================
def sample_speeds_grid(vmin: float, vmax: float, count: int) -> List[float]:
    """Evenly spaced speeds in [vmin, vmax]."""
    count = int(count)
    if count <= 1:
        return [float(vmin)]
    arr = np.linspace(float(vmin), float(vmax), count)
    return [float(x) for x in arr]


def sample_speeds_trunc_normal_random(
    rng: np.random.Generator,
    count: int,
    mean: float,
    std: float,
    vmin: float,
    vmax: float,
    max_tries: int = 200000,
) -> List[float]:
    """
    Random samples from truncated normal via rejection sampling.
    Has a max_tries guard to avoid infinite loops.
    """
    count = int(count)
    out: List[float] = []
    tries = 0
    while len(out) < count and tries < int(max_tries):
        x = float(rng.normal(loc=float(mean), scale=float(std)))
        if float(vmin) <= x <= float(vmax):
            out.append(x)
        tries += 1

    if len(out) < count:
        # fallback: clip extra samples
        extra = rng.normal(loc=float(mean), scale=float(std), size=(count - len(out)))
        extra = np.clip(extra, float(vmin), float(vmax))
        out.extend([float(x) for x in extra])
    return out


def sample_speeds_trunc_normal_quantiles(
    rng: np.random.Generator,
    count: int,
    mean: float,
    std: float,
    vmin: float,
    vmax: float,
    oversample: int = 50000,
) -> List[float]:
    """
    Stable representative points: generate many samples, truncate, then take quantiles.
    Good for "10 points" or "100 points" that represent the distribution.
    """
    count = int(count)
    xs = rng.normal(loc=float(mean), scale=float(std), size=int(oversample))
    xs = xs[(xs >= float(vmin)) & (xs <= float(vmax))]

    if xs.size < max(200, count * 10):
        return sample_speeds_trunc_normal_random(rng, count, mean, std, vmin, vmax)

    qs = np.linspace(0.0, 1.0, count + 2)[1:-1]  # exclude endpoints
    arr = np.quantile(xs, qs)
    return [float(x) for x in arr]


# =============================================================================
# Game MPC builder (build once, reuse)
# =============================================================================
def build_oracle_and_common_items(
    horizon_steps: int,
) -> Tuple[InteractionOracle, List[VehicleAction], List[VRUAction], PayoffWeights]:
    """Build shared objects used by all controllers."""
    AV = [VehicleAction(a) for a in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    AS = [VRUAction("yield"), VRUAction("go")]

    wv = PayoffWeights(
        w_collision_v=1e6,
        w_ttc_v=8e3,
        w_speed_v=6.0,
        w_acc_v=0.4,
        w_jerk_v=0.1,
    )

    oracle = InteractionOracle(horizon_steps=int(horizon_steps), vru_actions=AS)
    return oracle, AV, AS, wv


# =============================================================================
# Core simulation (no rendering)
# =============================================================================

VEH_INIT_X = -70.0
VEH_INIT_Y = -6.0
VEH_INIT_YAW = 0.0

def run_one_episode(
    cfg: EpisodeConfig,
    controller: str,
    max_steps: int,
    oracle: InteractionOracle,
    vehicle_actions: List[VehicleAction],
    weights_vehicle: PayoffWeights,
    v2x_range: float = 40.0,
    fov_range: float = 30.0,
    metric_mode: str = "end",  # "end" or "step"
) -> Dict[str, Any]:
    """
    metric_mode:
      - "end": compute evaluation once at end (recommended)
      - "step": compute evaluation every step (slow)
    """
    # ---- init vehicle
    veh = BicycleModel(
        params=BicycleModel_params,
        x=float(VEH_INIT_X),
        y=float(VEH_INIT_Y),
        psi=0.0,
        v=float(cfg.veh_speed),
        t0=0.0,
        beta=0.0,
        yaw=float(VEH_INIT_YAW),
    )
    delta_cmd = 0.0
    ref_speed = float(cfg.veh_speed)

    # for efficiency metric (reference path: straight line in x)
    ref_x = np.linspace(float(veh.x), 30.0, 300)
    ref_y = np.zeros_like(ref_x)

    # ---- init VRU
    vru = PointMassNewton(
        params=PointMass_params["escooter"],
        position=[float(cfg.vru_x0), float(cfg.vru_y0)],
        velocity=[0.0, 0.0],
        dt=veh.dt,
        vru_category="escooter",
        rider_type=cfg.rider_type,
    )

    path_mgr = VRUPathManager(
        stop_pt=(cfg.stop_x, cfg.stop_y),
        turn_pt=(cfg.turn_x, cfg.turn_y),
        dest_pt=(cfg.dest_x, cfg.dest_y),
        arrive_eps=1.2,
    )
    # --- belief initialization (Step-3 design) ---
    belief = RiderTypeBelief(p_aggr=0.5, alpha=1.0)
    SCENARIO_TAG = "intersection"
    TRUE_RIDER = cfg.rider_type

    # --- ego controller (fresh per-episode due to internal state for rule-based controllers) ---
    ego_ctrl = make_ego_controller(
        controller,
        oracle=oracle,
        vehicle_actions=vehicle_actions,
        weights_vehicle=weights_vehicle,
        verbose=False,
    )

    veh_sensor = VehFOV(
        fov_range=float(fov_range),
        fov_angle=2 * math.pi / 3,
        vehicle=veh,
        vru=vru,
    )
    vru_awr = VRU_AWR(profile=cfg.v2x_profile, init_state=1, seed=int(cfg.seed))

    metric = SustainEvaluator(
        start_pos=float(veh.x),
        finish_pos=30.0,
        veh=veh,
        desired_veh_velo=ref_speed,
        vru=vru,
    )

    # ---- episode stats
    first_detect_t: Optional[float] = None
    yield_count = 0
    first_yield_t: Optional[float] = None
    min_ttc = float("inf")

    end_reason = "max_steps"
    collision_flag = 0

    # ------------------- anti-deadlock params -------------------
    YIELD_FORCE_GO_STEPS = 60          # if VRU yields for 6s -> force "go"
    VEH_CREEP_AFTER_STEPS = 40         # if AV stuck while VRU yields for 4s -> creep
    V_STOP_EPS = 0.20                  # [m/s] consider stopped
    X_STUCK_EPS = 0.05                 # [m] no progress threshold
    SAFE_DIST_FOR_CREEP = 6.0          # [m] only creep if not too close
    CREEP_ACCEL = 1.0                  # [m/s^2] gentle creep forward
    DEADLOCK_STEPS = 100               # if fully stuck for 12s -> label as deadlock

    # ------------------- Rule-3 post-dynamics deadlock tracker (NEW) -------------------
    X_PROGRESS_EPS = 0.01  # [m] "actually moved this step" threshold (small so creep counts)
    deadlock_streak = 0  # counts consecutive steps with NO movement after dynamics

    # ------------------- override latches -------------------
    FORCE_GO_HOLD_STEPS = 25  # hold forced VRU "go" for 2.5s (dt=0.1)
    CREEP_HOLD_STEPS = 20  # hold forced AV creep accel for 2.0s
    YIELD_LOCK_STEPS = 30  # lock VRU in yield while AV creeps (3.0s)

    STOP_EPS = 1.2  # same as path_mgr arrive_eps
    VRU_STOP_SPD = 0.25  # [m/s] treat VRU as stopped at stop point

    vru_force_go_timer = 0
    veh_creep_timer = 0
    vru_yield_lock_timer = 0

    consecutive_yield = 0
    consecutive_stopped = 0
    forced_vru_go = 0
    forced_veh_creep = 0

    veh_x_prev = float(veh.x)
    vru_x_prev = float(vru.x)
    vru_y_prev = float(vru.y)


    # cached control
    a_cmd = 0.0
    vru_mode = "go"

    for step in range(int(max_steps)):
        # perception
        fov_flag = int(veh_sensor.update())
        link_state = int(vru_awr.link_check())

        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))
        v2x_detected = 1 if (link_state == 1 and dist_vru <= float(v2x_range)) else 0
        # Step-3: planning/belief driven ONLY by FOV detection; V2X logged only.
        detected_fov = bool(fov_flag == 1)
        detected = detected_fov

        if detected and first_detect_t is None:
            first_detect_t = float(veh.t + veh.dt)

        # --- hard code to allow veh react when it is risky scenario ---
        ttc_now = ttc_los(veh, vru)
        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))

        # threat gating (tune these)
        D_TH = 40.0  # meters
        TTC_TH = 6.0  # seconds

        threat = (dist_vru <= D_TH) and (ttc_now < TTC_TH)
        detected_for_planner = bool(detected and threat)

        # TTC stats: pre-dynamics snapshot (consistent with threat-gate timing)
        if math.isfinite(ttc_now):
            min_ttc = min(min_ttc, float(ttc_now))

        belief.update_from_motion(
            detected=bool(detected_fov),
            scenario=SCENARIO_TAG,
            vru_x=float(vru.x), vru_y=float(vru.y),
            vru_vx=float(vru.v_x), vru_vy=float(vru.v_y),
        )

        # ---- swappable controller call ----
        ctrl_out = ego_ctrl.plan(
            veh0=VehState.from_model(veh),
            vru0=VRUState.from_model(vru),
            delta0=delta_cmd,
            ref_speed=ref_speed,
            detected=bool(detected_for_planner),
            belief=belief,
            scenario=SCENARIO_TAG,
            vru_react_when_undetected=True,
        )

        a_cmd = float(getattr(ctrl_out, "a_cmd", 0.0))
        delta_cmd = float(getattr(ctrl_out, "delta_cmd", delta_cmd))

        dbg = getattr(ctrl_out, "debug", {}) or {}
        a0_by_type = dbg.get("a0_by_type", None)
        if isinstance(a0_by_type, dict) and (TRUE_RIDER in a0_by_type):
            a0 = a0_by_type[TRUE_RIDER]
            vru_mode = str(a0.mode) if hasattr(a0, "mode") else str(a0)
        else:
            vru_mode = str(dbg.get("vru_action_0", "go"))

        # =================== Yield lock until AV passes conflict (NEW) ===================
        # conflict x is the crossing line x0 (same as stop_x/turn_x in your configs)
        X_CONFLICT = float(cfg.stop_x)  # == x0
        PASS_X_MARGIN = 3.0  # meters, tune: 2~5
        STOP_EPS = 1.2  # same as path_mgr arrive_eps
        VRU_STOP_SPD = 0.25  # m/s

        # whether VRU is stopped at stop line
        vru_speed = math.hypot(float(vru.v_x), float(vru.v_y))
        near_stop = (math.hypot(float(vru.x) - float(cfg.stop_x), float(vru.y) - float(cfg.stop_y)) <= STOP_EPS)
        vru_stopped_at_stop = bool(near_stop and (vru_speed <= VRU_STOP_SPD))

        # whether AV has passed the conflict line (with some margin)
        veh_passed_conflict = bool(float(veh.x) >= X_CONFLICT + PASS_X_MARGIN)

        # If VRU is already yielding properly at stop line, it must KEEP yielding
        # until AV passes. This prevents "yield then go too late" collisions.
        if vru_stopped_at_stop and (not veh_passed_conflict):
            vru_mode = "yield"
        # ================================================================================

        # --- Late-yield guard ---
        # If pass stop line，yield is too late：force to go
        if vru_mode == "yield" and float(vru.y) > float(cfg.stop_y) + 0.5:
            vru_mode = "go"

        # ------------------- stop-point status (NEW) -------------------
        vru_speed = math.hypot(float(vru.v_x), float(vru.v_y))
        near_stop = (math.hypot(float(vru.x) - float(cfg.stop_x), float(vru.y) - float(cfg.stop_y)) <= STOP_EPS)
        vru_stopped_at_stop = bool(near_stop and (vru_speed <= VRU_STOP_SPD))

        # ------------------- apply latches (NEW) -------------------
        # Priority: (1) forced go, (2) yield lock, (3) vehicle creep
        if vru_force_go_timer > 0:
            vru_mode = "go"
            vru_force_go_timer -= 1

        if vru_yield_lock_timer > 0 and vru_force_go_timer <= 0:
            vru_mode = "yield"
            vru_yield_lock_timer -= 1

        if veh_creep_timer > 0:
            a_cmd = max(float(a_cmd), CREEP_ACCEL)
            veh_creep_timer -= 1

        # ------------------- anti-deadlock override -------------------
        # track yielding streak
        if vru_mode == "yield":
            consecutive_yield += 1
        else:
            consecutive_yield = 0

        # measure "stuck"
        veh_stopped = abs(float(veh.v)) < V_STOP_EPS
        veh_no_progress = abs(float(veh.x) - veh_x_prev) < X_STUCK_EPS
        vru_no_progress = (
            abs(float(vru.x) - vru_x_prev) < X_STUCK_EPS
            and abs(float(vru.y) - vru_y_prev) < X_STUCK_EPS
        )

        if veh_stopped and veh_no_progress and vru_no_progress:
            consecutive_stopped += 1
        else:
            consecutive_stopped = 0

        # distance guard
        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))

        # Rule 1 (UPDATED):
        # If VRU yields too long:
        #   - If VRU is yielding at the stop point (common intersection deadlock), do NOT force VRU to go.
        #     Instead, lock VRU in yield and let AV creep through.
        #   - Otherwise (VRU yielding in mid-interaction), force VRU go for a few seconds.
        if vru_mode == "yield" and consecutive_yield >= YIELD_FORCE_GO_STEPS:
            if vru_stopped_at_stop:
                # VRU is already yielding correctly -> AV should proceed
                veh_creep_timer = max(veh_creep_timer, CREEP_HOLD_STEPS)
                vru_yield_lock_timer = max(vru_yield_lock_timer, YIELD_LOCK_STEPS)
                forced_veh_creep += 1
            else:
                vru_force_go_timer = max(vru_force_go_timer, FORCE_GO_HOLD_STEPS)
                forced_vru_go += 1

            consecutive_yield = 0  # reset after intervention

        # Rule 2 (UPDATED):
        # If VRU is yielding and AV is stuck:
        #   - allow creep even when dist is small IF VRU is stopped at stop point
        #   - otherwise keep your original distance guard
        if vru_mode == "yield" and veh_stopped and veh_no_progress:
            if consecutive_yield >= VEH_CREEP_AFTER_STEPS:
                if vru_stopped_at_stop:
                    veh_creep_timer = max(veh_creep_timer, CREEP_HOLD_STEPS)
                    vru_yield_lock_timer = max(vru_yield_lock_timer, YIELD_LOCK_STEPS)
                    forced_veh_creep += 1
                else:
                    if dist_vru >= SAFE_DIST_FOR_CREEP:
                        veh_creep_timer = max(veh_creep_timer, CREEP_HOLD_STEPS)
                        forced_veh_creep += 1



        # ---- anti-deadlock override END-------

        if vru_mode == "yield":
            yield_count += 1
            if first_yield_t is None:
                first_yield_t = float(veh.t + veh.dt)

        # dynamics update (BEFORE) - snapshot positions for progress check (NEW)
        veh_x_old = float(veh.x)
        vru_x_old = float(vru.x)
        vru_y_old = float(vru.y)

        # dynamics update
        veh.update_state(a=a_cmd, delta=delta_cmd)

        target_xy, v_des = path_mgr.step_target(vru, vru_mode)
        Fx, Fy = sfm_force_to_target(vru, target_xy=target_xy, v_des=v_des, tau=0.6)
        vru.update(F_x=Fx, F_y=Fy)

        # ------------------- Rule-3: post-dynamics progress check -------------------
        veh_moved = abs(float(veh.x) - veh_x_old) >= X_PROGRESS_EPS
        vru_moved = (
                abs(float(vru.x) - vru_x_old) >= X_PROGRESS_EPS
                or abs(float(vru.y) - vru_y_old) >= X_PROGRESS_EPS
        )

        if veh_moved or vru_moved:
            deadlock_streak = 0
        else:
            deadlock_streak += 1

        # ---- anti-deadlock override -------
        # Rule 3 (UPDATED): if NO post-dynamics movement too long -> deadlock handling
        if deadlock_streak >= DEADLOCK_STEPS:
            # last-resort escape attempt before labeling deadlock
            if vru_stopped_at_stop:
                veh_creep_timer = max(veh_creep_timer, CREEP_HOLD_STEPS * 2)
                vru_yield_lock_timer = max(vru_yield_lock_timer, YIELD_LOCK_STEPS * 2)
                forced_veh_creep += 1
                deadlock_streak = 0  # reset and give another window
            else:
                end_reason = "deadlock"
                break
        # ---- anti-deadlock override Rule 3 end-------

        # collision
        col = check_collision(veh, vru)
        if col:
            collision_flag = 1

        # metrics (slow mode)
        if metric_mode == "step":
            metric.cal_eff_index(actual_finish_time=float(veh.t), ref_x=ref_x, ref_y=ref_y)
            metric.cal_ttc_ind(collision_state=bool(col))
            metric.cal_safety_index(v_impact=float(veh.v) if col else None)
            acc_traj = getattr(veh, "acc_traj", [])
            yaw_hist = getattr(veh, "yaw_hist", [])
            metric.cal_cmft_index(acc_traj=acc_traj,yaw_hist=yaw_hist)
            metric.cal_overall_index(coe_eff=1.0, coe_safety=1.0, coe_cmft=1.0)

        # previous position
        veh_x_prev = float(veh.x)
        vru_x_prev = float(vru.x)
        vru_y_prev = float(vru.y)

        # termination
        if col:
            end_reason = "collision"
            break
        if float(veh.x) >= 30.0:
            end_reason = "vehicle_finished"
            break

    # compute metrics once (recommended)
    if metric_mode != "step":
        metric.cal_eff_index(actual_finish_time=float(veh.t), ref_x=ref_x, ref_y=ref_y)
        metric.cal_ttc_ind(collision_state=bool(collision_flag))
        metric.cal_safety_index(v_impact=float(veh.v) if collision_flag else None)
        acc_traj = getattr(veh, "acc_traj", [])
        yaw_hist = getattr(veh, "yaw_hist", [])
        metric.cal_cmft_index(acc_traj=acc_traj,yaw_hist=yaw_hist)
        metric.cal_overall_index(coe_eff=1.0, coe_safety=1.0, coe_cmft=1.0)

    steps_done = step + 1
    min_ttc_out = "" if (not math.isfinite(min_ttc) or min_ttc == float("inf")) else float(min_ttc)

    return dict(
        controller=str(controller),
        collision=int(collision_flag),
        end_reason=end_reason,
        steps=int(steps_done),
        t_end=float(veh.t),
        first_detect_t=("" if first_detect_t is None else float(first_detect_t)),
        yield_count=int(yield_count),
        first_yield_t=("" if first_yield_t is None else float(first_yield_t)),
        min_ttc=min_ttc_out,
        eff=float(metric.eff_index),
        safety=float(metric.safety_index),
        comfort=float(metric.comfort_index),
        overall=float(metric.overall_index),
        # NEW: anti-deadlock bookkeeping
        forced_vru_go=int(forced_vru_go),
        forced_veh_creep=int(forced_veh_creep),
    )


# =============================================================================
# Excel writer
# =============================================================================
def write_xlsx(rows: List[List[Any]], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    wb = xwr.Workbook(out_path)
    ws = wb.add_worksheet("results")

    header_fmt = wb.add_format({"bold": True, "bg_color": "#EEEEEE"})
    avg_fmt = wb.add_format({"bold": True, "bg_color": "#FFF2CC"})

    min_ttc_col = rows[0].index("min_ttc")

    for r, row in enumerate(rows):
        if r == 0:
            for c, v in enumerate(row):
                ws.write(r, c, v, header_fmt)
            continue

        tag = row[min_ttc_col] if (len(row) > min_ttc_col) else ""
        is_avg = tag in ("AVG", "collision rate")

        fmt = avg_fmt if is_avg else None

        if fmt is None:
            ws.write_row(r, 0, row)
        else:
            for c, v in enumerate(row):
                ws.write(r, c, v, fmt)

    ws.freeze_panes(1, 0)
    wb.close()

# =============================================================================
# Excel-Summary writer
# =============================================================================
def write_xlsx_summary(rows: List[List[Any]], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    wb = xwr.Workbook(out_path)
    ws = wb.add_worksheet("summary")

    header_fmt = wb.add_format({"bold": True, "bg_color": "#EEEEEE"})
    num_fmt = wb.add_format({"num_format": "0.000000"})
    int_fmt = wb.add_format({"num_format": "0"})

    for r, row in enumerate(rows):
        if r == 0:
            for c, v in enumerate(row):
                ws.write(r, c, v, header_fmt)
            continue

        # Summary schema (Step-5):
        # [0] controller (text)
        # [1] episodes (int)
        # [2] collision count (int)
        # [3] collision rate (float)
        # [4] deadlock rate (float)
        # [5] yield count (int)
        # [6] forced_vru_go (int)
        # [7] forced_veh_creep (int)
        # [8:] mean scores (float)
        ws.write(r, 0, row[0])
        ws.write(r, 1, row[1], int_fmt)
        ws.write(r, 2, row[2], int_fmt)
        ws.write(r, 3, row[3], num_fmt)
        ws.write(r, 4, row[4], num_fmt)
        ws.write(r, 5, row[5], int_fmt)
        ws.write(r, 6, row[6], int_fmt)
        ws.write(r, 7, row[7], int_fmt)
        for c in range(8, len(row)):
            ws.write(r, c, row[c], num_fmt)

    ws.freeze_panes(1, 0)
    wb.close()

def write_csv(rows: List[List[Any]], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# =============================================================================
# Quant runner
# =============================================================================
def quant_run(
    out_dir: str,
    controllers: List[str],
    episode_cfgs: List[EpisodeConfig],
    max_steps: int,
    oracle: InteractionOracle,
    vehicle_actions: List[VehicleAction],
    weights_vehicle: PayoffWeights,
    metric_mode: str,
    print_every: int,
    eta_warmup: int,
    assume_sec_per_ep: float,
) -> str:
    header = [
        "run_id",
        "cfg_id",
        "controller",
        "v2x_profile",
        "rider_type",
        "vru_x0",
        "vru_y0",
        "stop_x",
        "stop_y",
        "turn_x",
        "turn_y",
        "dest_x",
        "dest_y",
        "veh_x0",
        "veh_y0",
        "veh_yaw0",
        "veh_speed",
        "seed",
        "collision",
        "end_reason",
        "steps",
        "t_end",
        "first_detect_t",
        "yield_count",
        "first_yield_t",
        "forced_vru_go", # NEW anti-deadlock bookkeeping
        "forced_veh_creep", # NEW anti-deadlock bookkeeping
        "min_ttc",
        "eff",
        "safety",
        "comfort",
        "overall",
    ]
    rows: List[List[Any]] = [header]

    summary_header = [
        "controller",
        "simulation episodes",
        "collision count",
        "collision rate",
        "deadlock rate",
        "yield count",
        "forced_vru_go",
        "forced_veh_creep",
        "efficiency score",
        "safety score",
        "comfort score",
        "overall score",
    ]
    summary_rows: List[List[Any]] = [summary_header]

    collision_header = [
        "run_id",
        "cfg_id",
        "controller",
        "v2x_profile",
        "rider_type",
        "veh_x0",
        "veh_y0",
        "veh_yaw0",
        "veh_speed",
        "seed",
        "vru_x0",
        "vru_y0",
        "stop_x",
        "stop_y",
        "turn_x",
        "turn_y",
        "dest_x",
        "dest_y",
        "end_reason",
        "steps",
        "t_end",
        "min_ttc",
    ]
    collision_rows: List[List[Any]] = [collision_header]

    total_eps = int(len(controllers) * len(episode_cfgs))
    print(f"[INFO] controllers={controllers}")
    print(f"[INFO] episodes_per_controller={len(episode_cfgs)}, total_runs={total_eps}")

    start_wall = time.perf_counter()
    done_all = 0
    run_id = 0

    for ctrl_name in controllers:
        print(f"\n[CTRL] {ctrl_name} | episodes={len(episode_cfgs)}", flush=True)

        ctrl_done = 0
        ctrl_collision = 0
        ctrl_deadlock = 0
        ctrl_yield_total = 0
        ctrl_forced_vru_go = 0
        ctrl_forced_veh_creep = 0

        ctrl_eff: List[float] = []
        ctrl_saf: List[float] = []
        ctrl_cmf: List[float] = []
        ctrl_ovl: List[float] = []

        coll_by_rider: Dict[str, int] = {}

        for cfg_id, cfg in enumerate(episode_cfgs):
            out = run_one_episode(
                cfg=cfg,
                controller=ctrl_name,
                max_steps=int(max_steps),
                oracle=oracle,
                vehicle_actions=vehicle_actions,
                weights_vehicle=weights_vehicle,
                metric_mode=str(metric_mode),
            )

            row = [
                    run_id,
                    int(cfg_id),
                    str(ctrl_name),
                    cfg.v2x_profile,
                    cfg.rider_type,
                    cfg.vru_x0,
                    cfg.vru_y0,
                    cfg.stop_x,
                    cfg.stop_y,
                    cfg.turn_x,
                    cfg.turn_y,
                    cfg.dest_x,
                    cfg.dest_y,
                    VEH_INIT_X,
                    VEH_INIT_Y,
                    VEH_INIT_YAW,
                    cfg.veh_speed,
                    cfg.seed,
                    out["collision"],
                    out["end_reason"],
                    out["steps"],
                    out["t_end"],
                    out["first_detect_t"],
                    out["yield_count"],
                    out["first_yield_t"],
                    out["forced_vru_go"], # anti-deadlock bookeeping
                    out["forced_veh_creep"], # anti-deadlock bookeeping
                    out["min_ttc"],
                    out["eff"],
                    out["safety"],
                    out["comfort"],
                    out["overall"],
                ]
            rows.append(row)

            ctrl_forced_vru_go += int(out.get("forced_vru_go", 0))
            ctrl_forced_veh_creep += int(out.get("forced_veh_creep", 0))
            if str(out.get("end_reason", "")) in ("deadlock", "max_steps"):
                ctrl_deadlock += 1

            if int(out.get("collision", 0)) == 1:
                collision_rows.append([
                        run_id,
                        int(cfg_id),
                        str(ctrl_name),
                        cfg.v2x_profile,
                        cfg.rider_type,
                        VEH_INIT_X,
                        VEH_INIT_Y,
                        VEH_INIT_YAW,
                        cfg.veh_speed,
                        cfg.seed,
                        cfg.vru_x0,
                        cfg.vru_y0,
                        cfg.stop_x,
                        cfg.stop_y,
                        cfg.turn_x,
                        cfg.turn_y,
                        cfg.dest_x,
                        cfg.dest_y,
                        out["end_reason"],
                        out["steps"],
                        out["t_end"],
                        out["min_ttc"],
                    ])

            ctrl_eff.append(float(out["eff"]))
            ctrl_saf.append(float(out["safety"]))
            ctrl_cmf.append(float(out["comfort"]))
            ctrl_ovl.append(float(out["overall"]))

            if int(out.get("collision", 0)) == 1:
                ctrl_collision += 1
                coll_by_rider[cfg.rider_type] = coll_by_rider.get(cfg.rider_type, 0) + 1

            ctrl_yield_total += int(out.get("yield_count", 0))

            run_id += 1
            done_all += 1
            ctrl_done += 1

            if (done_all == 1) or (print_every > 0 and done_all % int(print_every) == 0):
                elapsed = time.perf_counter() - start_wall
                if done_all < max(1, int(eta_warmup)):
                    avg_sec = float(assume_sec_per_ep)
                else:
                    avg_sec = float(elapsed / max(done_all, 1))
                eta_sec = avg_sec * max(total_eps - done_all, 0)

                print(
                    _progress_line_controller(
                        done_all=done_all,
                        total_all=total_eps,
                        controller=str(ctrl_name),
                        ctrl_done=ctrl_done,
                        ctrl_total=len(episode_cfgs),
                        avg_sec_per_ep=avg_sec,
                        eta_sec=eta_sec,
                    ),
                    flush=True,
                )

        denom = max(len(ctrl_eff), 1)
        collision_rate = float(ctrl_collision) / float(denom)
        deadlock_rate = float(ctrl_deadlock) / float(denom)

        summary_rows.append([
            str(ctrl_name),
            int(denom),
            int(ctrl_collision),
            float(collision_rate),
            float(deadlock_rate),
            int(ctrl_yield_total),
            int(ctrl_forced_vru_go),
            int(ctrl_forced_veh_creep),
            float(np.mean(ctrl_eff)) if ctrl_eff else 0.0,
            float(np.mean(ctrl_saf)) if ctrl_saf else 0.0,
            float(np.mean(ctrl_cmf)) if ctrl_cmf else 0.0,
            float(np.mean(ctrl_ovl)) if ctrl_ovl else 0.0,
        ])

        rider_breakdown = ", ".join([f"{k}:{v}" for k, v in sorted(coll_by_rider.items())]) or "none"
        print(
            f"[CTRL SUMMARY] {ctrl_name}: collision_rate={collision_rate:.3f} ({ctrl_collision}/{denom}), "
            f"deadlock_rate={deadlock_rate:.3f}, collisions_by_rider=({rider_breakdown})",
            flush=True,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) make a subfolder for this run
    run_dir = os.path.join(out_dir, f"quant_vei_intersection_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 2) write all outputs into this folder
    out_path = os.path.join(run_dir, f"quant_vei_intersection_details_{stamp}.xlsx")
    write_xlsx(rows, out_path)

    out_path_summary = os.path.join(run_dir, f"quant_vei_intersection_summary_{stamp}.xlsx")
    write_xlsx_summary(summary_rows, out_path_summary)

    out_path_coll_xlsx = os.path.join(run_dir, f"quant_vei_intersection_collisions_{stamp}.xlsx")
    write_xlsx(collision_rows, out_path_coll_xlsx)

    # out_path_coll_csv = os.path.join(run_dir, f"quant_vei_intersection_collisions_{stamp}.csv")
    # write_csv(collision_rows, out_path_coll_csv)

    print(f"[DONE] Saved all quant INTERSECTION result files to:\n  {run_dir}")
    print(f"  - {out_path}")
    print(f"  - {out_path_summary}")
    print(f"  - {out_path_coll_xlsx}")

    # Email notification — fires after all xlsx files are written.
    # summary_rows layout (skip header at index 0):
    #   [0]=controller, [1]=episodes, [2]=collisions, [9]=safety_score
    elapsed_sec = time.perf_counter() - start_wall
    ctrl_stats = [
        {
            "name":       str(row[0]),
            "collisions": int(row[2]),
            "total":      int(row[1]),
            "safety":     float(row[9]) if len(row) > 9 else float("nan"),
        }
        for row in summary_rows[1:]
    ]
    return out_path


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default=os.path.join("quant_sim", "quant_sim_results"))
    parser.add_argument("--max-steps", type=int, default=400)

    parser.add_argument("--v2x", type=str, default="no")
    parser.add_argument("--riders", type=str, default="Aggressive,Conservative")
    parser.add_argument("--seeds", type=str, default="0") # "0,1,2" if probability transition included

    parser.add_argument(
        "--controllers",
        type=str,
        default="rule1d,rule2d,mpc1d,mpc2d",
        help="Comma-separated controller list. Run sequentially.",
    )
    parser.add_argument(
        "--episodes-per-controller",
        type=int,
        default=10,
        help="Number of sampled configs per controller.",
    )

    # speed control (very explicit: 10 or 100 points)
    parser.add_argument("--speed-min", type=float, default=10.0)
    parser.add_argument("--speed-max", type=float, default=20.0)
    parser.add_argument("--speed-count", type=int, default=10, help="Number of speed points (e.g., 10 or 100).")
    parser.add_argument(
        "--speed-mode",
        type=str,
        choices=["grid", "normal_random", "normal_quantiles"],
        default="normal_quantiles",
    )
    parser.add_argument("--speed-mean", type=float, default=15.0)
    parser.add_argument("--speed-std", type=float, default=2.0)

    parser.add_argument("--rng-seed", type=int, default=123)

    # performance controls
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--metric-mode", type=str, choices=["end", "step"], default="end")

    # dry run
    parser.add_argument("--dry-run", action="store_true", help="Only print episode counts, do not run simulation.")

    # progress tracking
    parser.add_argument("--print-every", type=int, default=10, help="Print progress every N episodes. Use 0 to disable.")
    parser.add_argument("--eta-warmup", type=int, default=10, help="Use assume-sec-per-ep before warmup episodes are done.")
    parser.add_argument("--assume-sec-per-ep", type=float, default=2.0, help="Initial ETA uses this seconds/episode before warmup.")

    args = parser.parse_args()

    v2x_profiles = [s.strip() for s in args.v2x.split(",") if s.strip()]
    rider_types = [s.strip() for s in args.riders.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rng = np.random.default_rng(int(args.rng_seed))

    # speed sampling
    if args.speed_mode == "grid":
        speeds = sample_speeds_grid(args.speed_min, args.speed_max, args.speed_count)
    elif args.speed_mode == "normal_random":
        speeds = sample_speeds_trunc_normal_random(
            rng=rng,
            count=int(args.speed_count),
            mean=float(args.speed_mean),
            std=float(args.speed_std),
            vmin=float(args.speed_min),
            vmax=float(args.speed_max),
        )
    else:
        speeds = sample_speeds_trunc_normal_quantiles(
            rng=rng,
            count=int(args.speed_count),
            mean=float(args.speed_mean),
            std=float(args.speed_std),
            vmin=float(args.speed_min),
            vmax=float(args.speed_max),
        )

    # VRU configs
    vru_cfgs = build_vru_configs_default()

    total_cfg_space = _estimate_total_configs(
        v2x_profiles=v2x_profiles,
        rider_types=rider_types,
        seeds=seeds,
        speeds=speeds,
        vru_cfgs=vru_cfgs,
    )

    controllers = [s.strip() for s in args.controllers.split(",") if s.strip()]
    N = int(max(1, args.episodes_per_controller))
    total_runs = int(len(controllers) * N)

    est_total_sec = float(total_runs) * float(args.assume_sec_per_ep)
    print(f"[DRY INFO] config_space={total_cfg_space} possible episodes")
    print(f"[DRY INFO] controllers={controllers}")
    print(f"[DRY INFO] episodes_per_controller={N} => total_runs={total_runs}")
    print(f"[DRY INFO] ETA(using assume {args.assume_sec_per_ep:.2f}s/ep) ≈ {_fmt_seconds(est_total_sec)}")

    if args.dry_run:
        return

    oracle, AV, AS, wv = build_oracle_and_common_items(horizon_steps=int(args.horizon_steps))

    # sample configs ONCE, reuse across all controllers
    stream_all = (
        cfg
        for v2x in v2x_profiles
        for rider in rider_types
        for cfg in iter_episode_configs(v2x, rider, speeds, seeds, vru_cfgs)
    )
    episode_cfgs = reservoir_sample(stream_all, k=N, rng=rng)
    if len(episode_cfgs) < N:
        print(f"[WARN] Only {len(episode_cfgs)} unique configs available in config space.")

    tic = time.perf_counter()
    out_path = quant_run(
        out_dir=args.out_dir,
        controllers=controllers,
        episode_cfgs=episode_cfgs,
        max_steps=int(args.max_steps),
        oracle=oracle,
        vehicle_actions=AV,
        weights_vehicle=wv,
        metric_mode=str(args.metric_mode),
        print_every=int(args.print_every),
        eta_warmup=int(args.eta_warmup),
        assume_sec_per_ep=float(args.assume_sec_per_ep),
    )
    toc = time.perf_counter()

    print(f"[DONE] Total time: {toc - tic:.3f}s")


if __name__ == "__main__":
    main()
