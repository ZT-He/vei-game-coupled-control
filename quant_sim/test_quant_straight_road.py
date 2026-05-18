# quant_sim/test_quant_straight_road.py
"""Quantitative runner for VEI straight-road case (no rendering).

Goal
----
Run many episodes (configs) and save results to Excel.

Design requirement (critical)
----------------------------
The *single-episode simulation loop* must match `qual_sim/sim_vei_straight_road.py::run_simulation()`.
The ONLY difference is:
  - sim_vei renders one episode
  - test_quant runs many episodes and exports results

Therefore, this script:
  - Uses the same per-step ordering (FOV -> belief -> plan -> dynamics -> eval -> termination).
  - Uses the same fixed parameters as sim_vei (FOV range/angle, SFM tau, v2x_range, oracle horizon,
    action set, payoff weights, occluder traffic parameters).
  - Does NOT include any external override that changes the applied dynamics.

Update requested by user
------------------------
  - Add `veh_y0` as a configurable dimension in the episode config.
  - Support multiple `x_lc_start` options in `build_vru_configs_default()`.
"""

from __future__ import annotations

import os
import time
import math
import argparse
import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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

from interaction.oracle import InteractionOracle
from interaction.controllers.factory import make_ego_controller



# =============================================================================
# Fixed parameters (must match sim_vei_straight_road)
# =============================================================================

# Layout / finish
FINISH_X = 30.0

# Ego init (x/yaw fixed; y is configurable by cfg)
VEH_INIT_X = -80.0
VEH_INIT_YAW = 0.0

# Perception
FOV_RANGE = 30.0
FOV_ANGLE = 2.0 * math.pi / 3.0

# V2X (logged only; planning is driven by FOV only)
V2X_RANGE = 40.0

# SFM
SFM_TAU = 0.5

# Oracle / controllers
HORIZON_STEPS = 20
VEHICLE_ACTIONS = [VehicleAction(a) for a in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
VRU_ACTIONS = [VRUAction("yield"), VRUAction("go")]
WEIGHTS_VEHICLE = PayoffWeights(
    w_collision_v=1e6,
    w_ttc_v=8e3,
    w_speed_v=6.0,
    w_acc_v=0.4,
    w_jerk_v=0.1,
)

# Occluding traffic (same as sim)
TRAFFIC_SPEED = 8.0
TRAFFIC_OFFSET = 20.0
TRAFFIC_LANES_Y = (6.0, 2.0)

# Console alert: if collisions for one controller exceed this threshold, print a failure banner.
COLLISION_FAIL_THRESHOLD = 12

# =============================================================================
# Geometry helpers (occlusion polygons) – same convention as sim
# =============================================================================

def _vehicle_poly_at(
    x: float,
    y: float,
    yaw: float,
    *,
    length: float = 4.5,
    width: float = 2.0,
    c2r: float = 1.5,
) -> List[Tuple[float, float]]:
    """Rectangle polygon vertices in world frame."""
    X, Y, Yaw = float(x), float(y), float(yaw)
    L, W, C2R = float(length), float(width), float(c2r)

    offsets = np.array(
        [
            [-C2R, -W / 2],
            [-C2R, W / 2],
            [L - C2R, W / 2],
            [L - C2R, -W / 2],
        ]
    ).T  # 2x4

    Rm = np.array(
        [
            [math.cos(Yaw), -math.sin(Yaw)],
            [math.sin(Yaw), math.cos(Yaw)],
        ]
    )
    verts = (Rm @ offsets) + np.array([[X], [Y]])
    pts = verts.T.tolist()
    return [(float(px), float(py)) for px, py in pts]


@dataclass(frozen=True)
class TrafficVehicleCfg:
    x0: float
    y0: float
    v: float
    yaw: float = 0.0
    length: float = 4.5
    width: float = 2.0


def _moving_vehicle_poly_from_cfg(t: float, cfg: TrafficVehicleCfg, *, c2r: float = 1.5) -> List[Tuple[float, float]]:
    x = float(cfg.x0 + cfg.v * float(t))
    y = float(cfg.y0)
    return _vehicle_poly_at(x, y, float(cfg.yaw), length=float(cfg.length), width=float(cfg.width), c2r=float(c2r))


# =============================================================================
# Scenario VRU path manager (copied from sim_vei_straight_road)
# =============================================================================

class VRUPathManager:
    """Monotonic 3-stage lane-change manager (prevents turn-back loops)."""

    def __init__(
        self,
        y_start: float,
        y_target: float,
        x_lc_start: float,
        x_lc_done: float,
        *,
        lookahead: float = 12.0,
        y_eps: float = 0.6,
        v_go: float = 10.0,
        v_yield: float = 7.0,
        yield_max_steps: int = 60,
    ):
        self.y_start = float(y_start)
        self.y_target = float(y_target)
        self.x_lc_start = float(x_lc_start)
        self.x_lc_done = float(x_lc_done)

        self.lookahead = float(lookahead)
        self.y_eps = float(y_eps)

        self.v_go = float(v_go)
        self.v_yield = float(v_yield)

        self.stage = 0
        self._yield_streak = 0
        self._yield_max_steps = int(max(0, yield_max_steps))

    def step_target(self, vru, vru_mode: str):
        x = float(vru.x)
        y = float(vru.y)

        # anti-freeze: yielding too long -> force go
        if vru_mode == "yield":
            self._yield_streak += 1
            if self._yield_max_steps > 0 and self._yield_streak >= self._yield_max_steps:
                vru_mode = "go"
        else:
            self._yield_streak = 0

        # stage transitions
        if self.stage == 0 and x >= self.x_lc_start:
            self.stage = 1
        if self.stage == 1:
            if (abs(y - self.y_target) <= self.y_eps) and (x >= self.x_lc_done):
                self.stage = 2

        # reference y
        y_ref = self.y_start if self.stage == 0 else self.y_target

        # ALWAYS target ahead (prevents looping back)
        target_xy = (x + self.lookahead, y_ref)
        v_des = self.v_go if vru_mode == "go" else self.v_yield
        return target_xy, v_des


# =============================================================================
# Episode configuration
# =============================================================================

@dataclass(frozen=True)
class VRUConfig:
    vru_x0: float
    vru_y0: float
    y_start: float
    y_target: float
    x_lc_start: float
    x_lc_done: float
    lookahead: float
    y_eps: float


@dataclass(frozen=True)
class BaseEpisodeConfig:
    """A base scenario configuration that is shared across controllers and rider types."""

    # environment
    v2x_profile: str
    seed: int

    # ego
    veh_speed: float
    veh_y0: float

    # vru path
    vru_cfg: VRUConfig


def build_vru_configs_default(
    *,
    lane_pairs: Optional[List[Tuple[float, float]]] = None,
    x0_list: Optional[List[float]] = None,
    y_offset_list: Optional[List[float]] = None,
    x_lc_start_list: Optional[List[float]] = None,
    x_lc_done: float = 0.0,
    lookahead: float = 12.0,
    y_eps: float = 0.6,
) -> List[VRUConfig]:
    """Build VRU scenario configs for straight-road case.

    NOTE: default lane-pair matches sim_vei_straight_road: start at y=+6.0, target y=-2.0.
    """
    if lane_pairs is None:
        lane_pairs = [(6.0, -2.0)]
    if x0_list is None:
        x0_list = [-40.0, -35.0, -30.0, -25.0]
    if y_offset_list is None:
        y_offset_list = [0.0]
    if x_lc_start_list is None:
        # Requested: multiple x_lc_start options
        x_lc_start_list = [-25.0, -20.0, -15.0, -10.0, -5.0]

    out: List[VRUConfig] = []
    for (y_start, y_target) in lane_pairs:
        for x0 in x0_list:
            for dy in y_offset_list:
                for x_lc_start in x_lc_start_list:
                    vru_x0 = float(x0)
                    vru_y0 = float(y_start + dy)
                    out.append(
                        VRUConfig(
                            vru_x0=vru_x0,
                            vru_y0=vru_y0,
                            y_start=float(y_start),
                            y_target=float(y_target),
                            x_lc_start=float(x_lc_start),
                            x_lc_done=float(x_lc_done),
                            lookahead=float(lookahead),
                            y_eps=float(y_eps),
                        )
                    )
    return out


def _iter_base_episode_configs(
    *,
    v2x_profiles: List[str],
    seeds: List[int],
    veh_speeds: List[float],
    veh_y0_list: List[float],
    vru_cfgs: List[VRUConfig],
) -> Iterator[BaseEpisodeConfig]:
    for v2x in v2x_profiles:
        for sd in seeds:
            for vy0 in veh_y0_list:
                for spd in veh_speeds:
                    for vc in vru_cfgs:
                        yield BaseEpisodeConfig(
                            v2x_profile=str(v2x),
                            seed=int(sd),
                            veh_speed=float(spd),
                            veh_y0=float(vy0),
                            vru_cfg=vc,
                        )


def reservoir_sample(stream: Iterable[BaseEpisodeConfig], k: int, rng: np.random.Generator) -> List[BaseEpisodeConfig]:
    """Reservoir sampling: sample k items uniformly from a stream."""
    buf: List[BaseEpisodeConfig] = []
    for i, item in enumerate(stream):
        if i < k:
            buf.append(item)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                buf[j] = item
    return buf


# =============================================================================
# Misc helpers
# =============================================================================

def ttc_los(veh, vru) -> float:
    """Line-of-sight TTC. If not closing => inf."""
    rx = float(vru.x) - float(veh.x)
    ry = float(vru.y) - float(veh.y)
    dist = math.hypot(rx, ry)
    if dist < 1e-6:
        return 0.0

    vvx = float(veh.v) * math.cos(float(veh.yaw))
    vvy = float(veh.v) * math.sin(float(veh.yaw))

    rvx = float(vru.v_x) - vvx
    rvy = float(vru.v_y) - vvy

    closing = -(rx * rvx + ry * rvy) / dist
    if closing <= 1e-6:
        return float("inf")
    return dist / closing


def _call_belief_update(belief: RiderTypeBelief, *, debug: dict, **kwargs) -> None:
    """Call belief.update_from_motion with a debug kwarg only if supported (robust to versions)."""
    try:
        sig = inspect.signature(belief.update_from_motion)
        if "debug" in sig.parameters:
            kwargs["debug"] = debug
    except Exception:
        # If introspection fails, call without debug.
        pass
    belief.update_from_motion(**kwargs)


# =============================================================================
# Single-episode simulation (MUST match sim_vei_straight_road)
# =============================================================================

def run_one_episode(
    *,
    cfg: BaseEpisodeConfig,
    rider_type: str,
    ego_controller_name: str,
    ego_ctrl,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one episode with the same step logic as sim_vei_straight_road."""

    # ------------------- ego init (eastbound +x) -------------------
    veh = BicycleModel(
        params=BicycleModel_params,
        x=float(VEH_INIT_X),
        y=float(cfg.veh_y0),
        psi=0.0,
        v=float(cfg.veh_speed),
        t0=0.0,
        beta=0.0,
        yaw=float(VEH_INIT_YAW),
    )
    delta_cmd = 0.0
    ref_speed = float(cfg.veh_speed)

    # reference path for efficiency metric (match sim)
    ref_x = np.linspace(float(veh.x), float(FINISH_X), 300)
    ref_y = np.zeros_like(ref_x)

    # ------------------- VRU init -------------------
    vru = PointMassNewton(
        params=PointMass_params["escooter"],
        position=[float(cfg.vru_cfg.vru_x0), float(cfg.vru_cfg.vru_y0)],
        velocity=[0.0, 0.0],
        dt=veh.dt,
        vru_category="escooter",
        rider_type=str(rider_type),
    )

    # Match sim defaults (both types use v_go=10.0, v_yield=max(7, v_go-3)=7.0)
    vru_v_go = 10.0
    vru_v_yield = max(7.0, vru_v_go - 3.0)

    path_mgr = VRUPathManager(
        y_start=float(cfg.vru_cfg.y_start),
        y_target=float(cfg.vru_cfg.y_target),
        x_lc_start=float(cfg.vru_cfg.x_lc_start),
        x_lc_done=float(cfg.vru_cfg.x_lc_done),
        lookahead=float(cfg.vru_cfg.lookahead),
        y_eps=float(cfg.vru_cfg.y_eps),
        v_go=float(vru_v_go),
        v_yield=float(vru_v_yield),
        yield_max_steps=60,
    )

    # ------------------- belief -------------------
    belief = RiderTypeBelief(p_aggr=0.5, alpha=1.0)
    TRUE_RIDER = str(rider_type)
    SCENARIO_TAG = "straight_road"

    # ------------------- perception -------------------
    veh_sensor = VehFOV(
        fov_range=float(FOV_RANGE),
        fov_angle=float(FOV_ANGLE),
        vehicle=veh,
        vru=vru,
    )
    vru_awr = VRU_AWR(profile=str(cfg.v2x_profile), init_state=1, seed=int(cfg.seed))

    # ------------------- evaluator -------------------
    metric = SustainEvaluator(
        start_pos=float(veh.x),
        finish_pos=float(FINISH_X),
        veh=veh,
        desired_veh_velo=ref_speed,
        vru=vru,
    )

    # ------------------- traffic vehicles (occluders) -------------------
    traffic_x0 = float(cfg.vru_cfg.vru_x0) - float(TRAFFIC_OFFSET)
    traffic_cfgs = [
        TrafficVehicleCfg(x0=traffic_x0, y0=float(TRAFFIC_LANES_Y[0]), v=float(TRAFFIC_SPEED)),
        TrafficVehicleCfg(x0=traffic_x0, y0=float(TRAFFIC_LANES_Y[1]), v=float(TRAFFIC_SPEED)),
    ]

    # ------------------- episode stats -------------------
    collision_flag = 0
    end_reason = "max_steps"
    first_detect_t: Optional[float] = None
    first_yield_t: Optional[float] = None
    yield_count = 0
    min_ttc = float("inf")

    # ------------------- sim loop -------------------
    v2x_detected_last = 0
    fov_flag_last = 0

    for step in range(int(max_steps)):
        # 1) occlusion-aware FOV detection
        occluders = [_moving_vehicle_poly_from_cfg(t=float(veh.t), cfg=c) for c in traffic_cfgs]
        fov_flag = int(veh_sensor.update(occluder_polys=occluders))
        fov_flag_last = int(fov_flag)

        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))
        link_state = int(vru_awr.link_check())  # 0/1
        v2x_detected = 1 if (link_state == 1 and dist_vru <= float(V2X_RANGE)) else 0
        v2x_detected_last = int(v2x_detected)

        # Step-3 design: belief + planning are driven ONLY by FOV detection.
        detected = bool(fov_flag == 1)
        if detected and first_detect_t is None:
            first_detect_t = float(veh.t + veh.dt)

        # 2) belief update (motion cues, only when visually observed)
        belief_dbg: dict = {}
        _call_belief_update(
            belief,
            debug=belief_dbg,
            detected=bool(detected),
            scenario=SCENARIO_TAG,
            vru_x=float(vru.x),
            vru_y=float(vru.y),
            vru_vx=float(vru.v_x),
            vru_vy=float(vru.v_y),
            y_start=float(cfg.vru_cfg.y_start),
            y_target=float(cfg.vru_cfg.y_target),
        )

        # 3) ego controller (swappable)
        veh_state = VehState.from_model(veh)
        vru_state = VRUState.from_model(vru)
        ctrl_out = ego_ctrl.plan(
            veh0=veh_state,
            vru0=vru_state,
            delta0=float(delta_cmd),
            ref_speed=float(ref_speed),
            detected=bool(detected),
            belief=belief,
            scenario=SCENARIO_TAG,
            vru_react_when_undetected=True,
        )

        a_cmd = float(ctrl_out.a_cmd)
        delta_cmd = float(ctrl_out.delta_cmd)

        # use true rider type for simulation rollout (same selection rule as sim)
        a0_by_type = (ctrl_out.debug or {}).get("a0_by_type", None)
        if isinstance(a0_by_type, dict) and (TRUE_RIDER in a0_by_type):
            a0 = a0_by_type[TRUE_RIDER]
            vru_mode = str(a0.mode) if hasattr(a0, "mode") else str(a0)
        else:
            vru_mode = str((ctrl_out.debug or {}).get("vru_action_0", "go"))

        if vru_mode == "yield":
            yield_count += 1
            if first_yield_t is None:
                first_yield_t = float(veh.t + veh.dt)

        # 4) apply dynamics
        veh.update_state(a=a_cmd, delta=delta_cmd)
        target_xy, v_des = path_mgr.step_target(vru, vru_mode)
        Fx, Fy = sfm_force_to_target(vru, target_xy=target_xy, v_des=v_des, tau=float(SFM_TAU))
        vru.update(F_x=Fx, F_y=Fy)

        # TTC stats
        ttc_now = ttc_los(veh, vru)
        if math.isfinite(ttc_now):
            min_ttc = min(min_ttc, float(ttc_now))

        # 5) collision + evaluation
        col = bool(check_collision(veh, vru))
        if col:
            collision_flag = 1

        metric.cal_eff_index(actual_finish_time=float(veh.t), ref_x=ref_x, ref_y=ref_y)
        metric.cal_ttc_ind(collision_state=bool(col))
        metric.cal_safety_index(v_impact=float(veh.v) if col else None)

        metric.jerk_traj = []
        metric.comfort_index_traj = []
        metric.cal_cmft_index(acc_traj=veh.acc_traj, yaw_hist=veh.yaw_hist)
        metric.cal_overall_index(coe_eff=1.0, coe_safety=1.0, coe_cmft=1.0)

        # 6) break conditions
        if col:
            end_reason = "collision"
            break
        if float(veh.x) >= float(FINISH_X):
            end_reason = "vehicle_finished"
            break

    steps_done = int(step + 1)
    min_ttc_out: Any = "" if (not math.isfinite(min_ttc) or min_ttc == float("inf")) else float(min_ttc)

    return dict(
        controller=str(ego_controller_name),
        rider_type=str(rider_type),
        v2x_profile=str(cfg.v2x_profile),
        seed=int(cfg.seed),
        veh_x0=float(VEH_INIT_X),
        veh_y0=float(cfg.veh_y0),
        veh_yaw0=float(VEH_INIT_YAW),
        veh_speed=float(cfg.veh_speed),
        vru_x0=float(cfg.vru_cfg.vru_x0),
        vru_y0=float(cfg.vru_cfg.vru_y0),
        y_start=float(cfg.vru_cfg.y_start),
        y_target=float(cfg.vru_cfg.y_target),
        x_lc_start=float(cfg.vru_cfg.x_lc_start),
        x_lc_done=float(cfg.vru_cfg.x_lc_done),
        lookahead=float(cfg.vru_cfg.lookahead),
        y_eps=float(cfg.vru_cfg.y_eps),
        collision=int(collision_flag),
        end_reason=str(end_reason),
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
        # last-step visibility flags (debug)
        fov=int(fov_flag_last),
        v2x=int(v2x_detected_last),
        p_aggr=float(getattr(belief, "p_aggr", float("nan"))),
    )


# =============================================================================
# Excel writers
# =============================================================================

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_xlsx(rows: List[List[Any]], out_path: str, sheet_name: str) -> None:
    _ensure_dir(out_path)
    wb = xwr.Workbook(out_path)
    ws = wb.add_worksheet(sheet_name)
    header_fmt = wb.add_format({"bold": True, "bg_color": "#EEEEEE"})

    for r, row in enumerate(rows):
        if r == 0:
            for c, v in enumerate(row):
                ws.write(r, c, v, header_fmt)
        else:
            ws.write_row(r, 0, row)
    ws.freeze_panes(1, 0)
    wb.close()


def write_xlsx_summary(rows: List[List[Any]], out_path: str) -> None:
    _ensure_dir(out_path)
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

        # controller as text
        ws.write(r, 0, row[0])

        # integer count columns
        for c in range(1, 5):  # episodes, collisions, deadlocks_or_max_steps, forced_vru_go
            ws.write(r, c, int(row[c]), int_fmt)

        # rates + means (floats)
        for c in range(5, len(row)):
            ws.write(r, c, float(row[c]), num_fmt)

    ws.freeze_panes(1, 0)
    wb.close()



# =============================================================================
# Quant runner
# =============================================================================

def quant_run(
    *,
    out_dir: str,
    controllers: List[str],
    rider_types: List[str],
    base_cfgs: List[BaseEpisodeConfig],
    max_steps: int,
    horizon_steps: int,
) -> str:
    """Run the cartesian product: base_cfg × controller × rider_type.

    Notes
    -----
    - Detail episode ordering follows the loop ordering:
        controller (outer) -> base_cfg -> rider_type (inner)
      so it matches the user's expectation.
    - Summary is ONE row per controller (Aggressive+Conservative combined),
      in the same controller order as the details sheet.
    """

    oracle = InteractionOracle(horizon_steps=int(horizon_steps), vru_actions=VRU_ACTIONS)

    # details rows
    header = [
        "run_id",
        "base_cfg_id",
        "controller",
        "rider_type",
        "v2x_profile",
        "seed",
        "veh_x0",
        "veh_y0",
        "veh_yaw0",
        "veh_speed",
        "vru_x0",
        "vru_y0",
        "y_start",
        "y_target",
        "x_lc_start",
        "x_lc_done",
        "lookahead",
        "y_eps",
        "collision",
        "end_reason",
        "steps",
        "t_end",
        "first_detect_t",
        "yield_count",
        "first_yield_t",
        "min_ttc",
        "eff",
        "safety",
        "comfort",
        "overall",
        "p_aggr_final",
        "fov_last",
        "v2x_last",
    ]
    details_rows: List[List[Any]] = [header]
    collision_rows: List[List[Any]] = [header]

    # summary: ONE row per controller (rider types combined)
    summary_header = [
        "controller",
        "episodes",
        "collisions",
        "deadlocks_or_max_steps",
        "forced_vru_go",
        "collision_rate",
        "deadlock_rate",
        "eff_mean",
        "safety_mean",
        "comfort_mean",
        "overall_mean",
        "yield_count_mean",
    ]
    summary_rows: List[List[Any]] = [summary_header]

    total_runs = len(base_cfgs) * len(controllers) * len(rider_types)
    run_id = 0
    t0 = time.perf_counter()

    for ctrl_name in controllers:
        # NOTE: create a fresh controller instance for every episode to avoid
        # cross-episode state leakage (e.g., latched modes, timers, MPC contracts).
        # This keeps single-episode behavior consistent with sim_vei.

        # per-controller aggregators (combined across rider types)
        ctrl_n = 0
        ctrl_collisions = 0
        ctrl_deadlocks = 0  # deadlock := episode terminates by reaching max_steps
        ctrl_forced_vru_go = 0  # kept for compatibility; no external override => always 0

        ctrl_eff: List[float] = []
        ctrl_safety: List[float] = []
        ctrl_comfort: List[float] = []
        ctrl_overall: List[float] = []
        ctrl_yield_count: List[float] = []

        ctrl_collisions_by_type: Dict[str, int] = {str(rt): 0 for rt in rider_types}

        ctrl_fail_printed = False

        for base_cfg_id, base_cfg in enumerate(base_cfgs):
            for rider_type in rider_types:
                # Reset controller state: instantiate a fresh controller per episode
                ego_ctrl = make_ego_controller(
                    str(ctrl_name),
                    oracle=oracle,
                    vehicle_actions=VEHICLE_ACTIONS,
                    weights_vehicle=WEIGHTS_VEHICLE,
                    verbose=False,
                )

                out = run_one_episode(
                    cfg=base_cfg,
                    rider_type=str(rider_type),
                    ego_controller_name=str(ctrl_name),
                    ego_ctrl=ego_ctrl,
                    max_steps=int(max_steps),
                )

                row = [
                    run_id,
                    base_cfg_id,
                    out["controller"],
                    out["rider_type"],
                    out["v2x_profile"],
                    out["seed"],
                    out["veh_x0"],
                    out["veh_y0"],
                    out["veh_yaw0"],
                    out["veh_speed"],
                    out["vru_x0"],
                    out["vru_y0"],
                    out["y_start"],
                    out["y_target"],
                    out["x_lc_start"],
                    out["x_lc_done"],
                    out["lookahead"],
                    out["y_eps"],
                    out["collision"],
                    out["end_reason"],
                    out["steps"],
                    out["t_end"],
                    out["first_detect_t"],
                    out["yield_count"],
                    out["first_yield_t"],
                    out["min_ttc"],
                    out["eff"],
                    out["safety"],
                    out["comfort"],
                    out["overall"],
                    out["p_aggr"],
                    out["fov"],
                    out["v2x"],
                ]
                details_rows.append(row)
                if int(out["collision"]) == 1:
                    collision_rows.append(row)

                # ---- per-controller summary stats ----
                ctrl_n += 1
                is_collision = int(out["collision"]) == 1
                if is_collision:
                    ctrl_collisions += 1

                    # Print running collision count for this controller (does NOT affect simulation).
                    print(f"[COLLISION] ctrl={ctrl_name} collision number={ctrl_collisions}", flush=True)
                    if (not ctrl_fail_printed) and (int(ctrl_collisions) > int(COLLISION_FAIL_THRESHOLD)):
                        print(
                            f"######################",
                            f"######{ctrl_name} is more than {COLLISION_FAIL_THRESHOLD} and fails!######",
                            f"######################",
                            flush=True,
                        )
                        ctrl_fail_printed = True


                    ctrl_collisions_by_type[str(rider_type)] = ctrl_collisions_by_type.get(str(rider_type), 0) + 1

                if str(out.get("end_reason", "")) == "max_steps":
                    ctrl_deadlocks += 1

                ctrl_eff.append(float(out["eff"]))
                ctrl_safety.append(float(out["safety"]))
                ctrl_comfort.append(float(out["comfort"]))
                ctrl_overall.append(float(out["overall"]))
                ctrl_yield_count.append(float(out["yield_count"]))

                run_id += 1
                if run_id % 10 == 0 or run_id == 1:
                    elapsed = time.perf_counter() - t0
                    avg = elapsed / max(run_id, 1)
                    eta = avg * max(total_runs - run_id, 0)
                    print(
                        f"[PROGRESS] {run_id}/{total_runs} | avg={avg:.2f}s/run | ETA≈{eta/60.0:.1f}min",
                        flush=True,
                    )

        # ---- finish one controller: compute summary row + print stats ----
        denom = max(int(ctrl_n), 1)
        collision_rate = float(ctrl_collisions) / float(denom)
        deadlock_rate = float(ctrl_deadlocks) / float(denom)

        summary_rows.append(
            [
                str(ctrl_name),
                int(ctrl_n),
                int(ctrl_collisions),
                int(ctrl_deadlocks),
                int(ctrl_forced_vru_go),
                float(collision_rate),
                float(deadlock_rate),
                float(np.mean(ctrl_eff)) if ctrl_eff else float("nan"),
                float(np.mean(ctrl_safety)) if ctrl_safety else float("nan"),
                float(np.mean(ctrl_comfort)) if ctrl_comfort else float("nan"),
                float(np.mean(ctrl_overall)) if ctrl_overall else float("nan"),
                float(np.mean(ctrl_yield_count)) if ctrl_yield_count else float("nan"),
            ]
        )

        # required console printout
        coll_aggr = int(ctrl_collisions_by_type.get("Aggressive", 0))
        coll_cons = int(ctrl_collisions_by_type.get("Conservative", 0))
        print(
            f"[CTRL DONE] {ctrl_name}: episodes={ctrl_n} | collision number={ctrl_collisions} | collision rate ={collision_rate} "
            f"(Aggressive={coll_aggr}, Conservative={coll_cons}) | "
            f"deadlocks(max_steps)={ctrl_deadlocks} (rate={deadlock_rate:.6f})",
            flush=True,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"quant_vei_straight_road_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    details_path = os.path.join(run_dir, f"details_{stamp}.xlsx")
    collisions_path = os.path.join(run_dir, f"collisions_{stamp}.xlsx")
    summary_path = os.path.join(run_dir, f"summary_{stamp}.xlsx")

    write_xlsx(details_rows, details_path, sheet_name="details")
    write_xlsx(collision_rows, collisions_path, sheet_name="collisions")
    write_xlsx_summary(summary_rows, summary_path)

    print(f"\n[DONE] Saved results to: {run_dir}")
    print(f"  - {details_path}")
    print(f"  - {collisions_path}")
    print(f"  - {summary_path}")

    # Email notification — fires after all xlsx files are written.
    # summary_rows layout (skip header at index 0):
    #   [0]=controller, [1]=episodes, [2]=collisions, [8]=safety_mean
    elapsed_sec = time.perf_counter() - t0
    ctrl_stats = [
        {
            "name":       str(row[0]),
            "collisions": int(row[2]),
            "total":      int(row[1]),
            "safety":     float(row[8]) if len(row) > 8 else float("nan"),
        }
        for row in summary_rows[1:]
    ]
    return run_dir



# =============================================================================
# Speed sampling (kept from your original quant script)
# =============================================================================

def sample_speeds_grid(vmin: float, vmax: float, count: int) -> List[float]:
    count = int(count)
    if count <= 1:
        return [float(vmin)]
    arr = np.linspace(float(vmin), float(vmax), count)
    return [float(x) for x in arr]


def sample_speeds_trunc_normal_quantiles(
    rng: np.random.Generator,
    count: int,
    mean: float,
    std: float,
    vmin: float,
    vmax: float,
    oversample: int = 50000,
) -> List[float]:
    count = int(count)
    xs = rng.normal(loc=float(mean), scale=float(std), size=int(oversample))
    xs = xs[(xs >= float(vmin)) & (xs <= float(vmax))]
    if xs.size < max(200, count * 10):
        arr = np.clip(rng.normal(loc=float(mean), scale=float(std), size=count), float(vmin), float(vmax))
        return [float(x) for x in arr]
    qs = np.linspace(0.0, 1.0, count + 2)[1:-1]
    arr = np.quantile(xs, qs)
    return [float(x) for x in arr]


# =============================================================================
# CLI
# =============================================================================

def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default=os.path.join("quant_sim", "quant_sim_results"))
    parser.add_argument("--max-steps", type=int, default=300)

    parser.add_argument("--controllers", type=str, default="rule1d,rule2d,mpc1d,mpc2d")
    parser.add_argument("--riders", type=str, default="Aggressive,Conservative")

    parser.add_argument("--v2x", type=str, default="no", help="Comma list: no,bad,good,perfect")
    parser.add_argument("--seeds", type=str, default="0")

    # ego initial lateral position choices
    parser.add_argument(
        "--veh-y0-list",
        type=str,
        default="-2.0",
        help="Comma list of ego initial lane centers (e.g., '-2.0,-6.0').",
    )

    # speed sampling
    parser.add_argument("--speed-min", type=float, default=10.0)
    parser.add_argument("--speed-max", type=float, default=20.0)
    parser.add_argument("--speed-count", type=int, default=100)
    parser.add_argument("--speed-mode", type=str, choices=["grid", "normal_quantiles"], default="normal_quantiles")
    parser.add_argument("--speed-mean", type=float, default=15.0)
    parser.add_argument("--speed-std", type=float, default=2.0)

    # VRU config
    parser.add_argument(
        "--x-lc-start-list",
        type=str,
        default="-15,-10,-5",
        help="Comma list of x_lc_start candidates.",
    )
    parser.add_argument("--x-lc-done", type=float, default=0.0)
    parser.add_argument("--vru-x0-list", type=str, default="-35,-30,-25")
    parser.add_argument("--lane-pairs", type=str, default="6:-6", help="Comma list like '6:-2,6:-6'")
    parser.add_argument("--lookahead", type=float, default=12.0)
    parser.add_argument("--y-eps", type=float, default=0.6)

    # sampling base configs
    parser.add_argument(
        "--num-base-cfg",
        type=int,
        default=10,
        help="How many base configs to sample (total runs = base_cfg * controllers * riders).",
    )
    parser.add_argument("--rng-seed", type=int, default=123)
    parser.add_argument("--dry-run", action="store_true")

    # oracle horizon (keep default = sim)
    parser.add_argument("--horizon-steps", type=int, default=int(HORIZON_STEPS))

    args = parser.parse_args()

    controllers = [s.strip() for s in str(args.controllers).split(",") if s.strip()]
    rider_types = [s.strip() for s in str(args.riders).split(",") if s.strip()]
    v2x_profiles = [s.strip() for s in str(args.v2x).split(",") if s.strip()]
    seeds = _parse_int_list(args.seeds)
    veh_y0_list = _parse_float_list(args.veh_y0_list)

    rng = np.random.default_rng(int(args.rng_seed))
    if args.speed_mode == "grid":
        veh_speeds = sample_speeds_grid(args.speed_min, args.speed_max, int(args.speed_count))
    else:
        veh_speeds = sample_speeds_trunc_normal_quantiles(
            rng=rng,
            count=int(args.speed_count),
            mean=float(args.speed_mean),
            std=float(args.speed_std),
            vmin=float(args.speed_min),
            vmax=float(args.speed_max),
        )

    # parse lane pairs like "6:-2,6:-6"
    lane_pairs: List[Tuple[float, float]] = []
    for item in str(args.lane_pairs).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad --lane-pairs item: '{item}' (expected 'yStart:yTarget')")
        a, b = item.split(":", 1)
        lane_pairs.append((float(a), float(b)))

    vru_x0_list = _parse_float_list(args.vru_x0_list)
    x_lc_start_list = _parse_float_list(args.x_lc_start_list)

    vru_cfgs = build_vru_configs_default(
        lane_pairs=lane_pairs,
        x0_list=vru_x0_list,
        x_lc_start_list=x_lc_start_list,
        x_lc_done=float(args.x_lc_done),
        lookahead=float(args.lookahead),
        y_eps=float(args.y_eps),
    )

    # Build stream then sample base cfgs once, reused for all controllers and rider types.
    stream = _iter_base_episode_configs(
        v2x_profiles=v2x_profiles,
        seeds=seeds,
        veh_speeds=veh_speeds,
        veh_y0_list=veh_y0_list,
        vru_cfgs=vru_cfgs,
    )

    # total possible combos (for informational purposes)
    total_possible = len(v2x_profiles) * len(seeds) * len(veh_y0_list) * len(veh_speeds) * len(vru_cfgs)
    k = int(min(max(1, int(args.num_base_cfg)), max(1, total_possible)))
    base_cfgs = reservoir_sample(stream, k=k, rng=rng)

    total_runs = len(base_cfgs) * len(controllers) * len(rider_types)
    print(
        f"[INFO] total_possible_base_cfg={total_possible} | sampled_base_cfg={len(base_cfgs)}\n"
        f"[INFO] controllers={controllers}\n"
        f"[INFO] riders={rider_types}\n"
        f"[INFO] total_runs = {len(base_cfgs)} * {len(controllers)} * {len(rider_types)} = {total_runs}",
        flush=True,
    )

    if args.dry_run:
        print("[DRY RUN] Not running simulations.")
        return

    tic = time.perf_counter()
    quant_run(
        out_dir=str(args.out_dir),
        controllers=controllers,
        rider_types=rider_types,
        base_cfgs=base_cfgs,
        max_steps=int(args.max_steps),
        horizon_steps=int(args.horizon_steps),
    )

    toc = time.perf_counter()
    print(f"[DONE] Total time: {toc - tic:.3f}s")


if __name__ == "__main__":
    main()

