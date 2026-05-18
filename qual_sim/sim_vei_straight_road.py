# qual_sim/sim_vei_straight_road.py
from __future__ import annotations

import os
import subprocess
import math
import time
import argparse
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from traffic_env.traffic_layout import TrafficLayout
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

from qual_sim.render_vei_straight_road import VEIStraightRoadRenderer, TrafficVehicleCfg

# ----------------------- occlusion polygons for FOV -----------------------

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

    offsets = np.array([
        [-C2R, -W / 2],
        [-C2R,  W / 2],
        [ L - C2R,  W / 2],
        [ L - C2R, -W / 2],
    ]).T  # 2x4

    Rm = np.array([
        [math.cos(Yaw), -math.sin(Yaw)],
        [math.sin(Yaw),  math.cos(Yaw)],
    ])
    verts = (Rm @ offsets) + np.array([[X], [Y]])
    pts = verts.T.tolist()
    return [(float(px), float(py)) for px, py in pts]


def _moving_vehicle_poly_from_cfg(t: float, cfg: TrafficVehicleCfg, *, c2r: float = 1.5) -> List[Tuple[float, float]]:
    x = float(cfg.x0 + cfg.v * float(t))
    y = float(cfg.y0)
    return _vehicle_poly_at(x, y, float(cfg.yaw), length=float(cfg.length), width=float(cfg.width), c2r=float(c2r))


# ----------------------- scenario VRU path manager -----------------------

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
        if self.stage == 0:
            y_ref = self.y_start
        else:
            y_ref = self.y_target

        # ALWAYS target ahead (prevents looping back)
        target_xy = (x + self.lookahead, y_ref)
        v_des = self.v_go if vru_mode == "go" else self.v_yield
        return target_xy, v_des


def run_simulation(
    v2x_profile: str = "no",
    rider_type: str = "Conservative",
    ego_controller: str = "mpc1d",
    max_steps: int = 300,
    render: bool = True,
    save_frames: bool = True,
    out_dir: Optional[str] = None,
    seed: int = 42,
    veh_speed: float = 14.58,
    veh_y0: float = -2.0,
    vru_x0: float = -30.0,
    vru_y0: float = 6.0,
    y_start: float = 6.0,
    y_target: float = -6.0,
    x_lc_start: float = -15.0,
    x_lc_done: float = 0.0,
    lookahead: float = 12.0,
    y_eps: float = 0.6,
):
    # ------------------- output dir -------------------
    if out_dir is None:
        out_dir = os.path.join("qual_sim", "output", "straight_road")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------- layout -------------------
    layout = TrafficLayout(
        "vru_euroncap_strrd",
        X_min=-100, X_max=100,
        Y_min=-35, Y_max=35,
        W_lane=4.0, num_lanes=4,
    )

    # ------------------- ego vehicle init (eastbound +x) -------------------

    ref_speed = float(veh_speed)

    veh = BicycleModel(
        params=BicycleModel_params,
        x=-80.0,
        y=float(veh_y0),
        psi=0.0,
        v=ref_speed,
        t0=0.0,
        beta=0.0,
        yaw=0.0,
    )
    delta_cmd = 0.0

    # reference path for efficiency metric
    ref_x = np.linspace(float(veh.x), 30.0, 300)
    ref_y = np.zeros_like(ref_x)

    # ------------------- VRU init -------------------
    vru = PointMassNewton(
        params=PointMass_params["escooter"],
        position=[float(vru_x0), float(vru_y0)],
        velocity=[0.0, 0.0],
        dt=veh.dt,
        vru_category="escooter",
        rider_type=rider_type,
    )

    START_LANE_Y = float(y_start)
    TARGET_LANE_Y = float(y_target)

    vru_v_go = 10.0
    vru_v_yield = max(7.0, vru_v_go - 3.0)

    path_mgr = VRUPathManager(
        y_start=START_LANE_Y,
        y_target=TARGET_LANE_Y,
        x_lc_start=float(x_lc_start),
        x_lc_done=float(x_lc_done),
        lookahead=float(lookahead),
        y_eps=float(y_eps),
        v_go=vru_v_go,
        v_yield=vru_v_yield,
        yield_max_steps=60,
    )

    # ------------------- belief (Step 3 default) -------------------
    belief = RiderTypeBelief(p_aggr=0.5, alpha=1.0)
    TRUE_RIDER = rider_type
    SCENARIO_TAG = "straight_road"

    # ------------------- perception -------------------
    veh_sensor = VehFOV(
        fov_range=30.0,
        fov_angle=2 * math.pi / 3,
        vehicle=veh,
        vru=vru,
    )

    # V2X is *logged* only. Planning uses FOV only (per Step 3 design).
    vru_awr = VRU_AWR(profile=v2x_profile, init_state=1, seed=seed)
    v2x_range = 40.0

    # ------------------- interaction oracle + swappable ego controller (Step 5) -------------------
    AV = [VehicleAction(a) for a in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    AS = [VRUAction("yield"), VRUAction("go")]

    wv = PayoffWeights(
        w_collision_v=1e6,
        w_ttc_v=8e3,
        w_speed_v=6.0,
        w_acc_v=0.4,
        w_jerk_v=0.1,
    )

    oracle = InteractionOracle(horizon_steps=20, vru_actions=AS)
    ego_ctrl = make_ego_controller(
        ego_controller,
        oracle=oracle,
        vehicle_actions=AV,
        weights_vehicle=wv,
        verbose=False,
    )

    # ------------------- evaluator -------------------
    metric = SustainEvaluator(
        start_pos=float(veh.x),
        finish_pos=30.0,
        veh=veh,
        desired_veh_velo=ref_speed,
        vru=vru,
    )

    # ------------------- logs -------------------
    log = {
        "t": [],
        "veh_x": [], "veh_v": [], "veh_a": [], "veh_delta": [],
        "vru_x": [], "vru_y": [], "vru_v": [],
        "fov": [], "v2x": [], "detected": [],
        "vru_mode": [], "collision": [],
        "p_aggr": [], "go_score_s": [], "p_go_obs": [],
        "safety": [], "eff": [], "comfort": [], "overall": [],
    }

    ttc_bin_hist: List[Optional[int]] = []

    # ------------------- traffic vehicles (viz + occluders) -------------------
    TRAFFIC_SPEED = 8.0
    TRAFFIC_OFFSET = 20.0
    traffic_x0 = float(vru_x0) - float(TRAFFIC_OFFSET)

    traffic_cfgs = [
        TrafficVehicleCfg(x0=traffic_x0, y0=6.0, v=TRAFFIC_SPEED),
        TrafficVehicleCfg(x0=traffic_x0, y0=2.0, v=TRAFFIC_SPEED),
    ]

    # ------------------- render init -------------------
    renderer = None
    if render:
        plt.ion()
        renderer = VEIStraightRoadRenderer(
            layout=layout,
            out_dir=out_dir,
            xlim=(-50, 50),
            ylim=(-50, 50),
            tail_len=30,
            dpi=160,
            save_frames=save_frames,
            traffic=traffic_cfgs,
            show_wifi=True,
        )

    # ------------------- sim loop -------------------
    for step in range(int(max_steps)):
        # 1) occlusion-aware FOV detection
        occluders = [_moving_vehicle_poly_from_cfg(t=float(veh.t), cfg=cfg) for cfg in traffic_cfgs]
        fov_flag = int(veh_sensor.update(occluder_polys=occluders))

        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))
        link_state = int(vru_awr.link_check())  # 0/1
        v2x_detected = 1 if (link_state == 1 and dist_vru <= v2x_range) else 0

        # Step-3 design: belief + planning are driven ONLY by FOV detection.
        detected = bool(fov_flag == 1)

        # 2) belief update (motion cues, only when visually observed)
        belief_dbg: dict = {}
        belief.update_from_motion(
            detected=bool(detected),
            scenario=SCENARIO_TAG,
            vru_x=float(vru.x), vru_y=float(vru.y),
            vru_vx=float(vru.v_x), vru_vy=float(vru.v_y),
            y_start=START_LANE_Y,
            y_target=TARGET_LANE_Y,
            debug=belief_dbg,
        )

        # 3) ego controller (swappable)
        veh_state = VehState.from_model(veh)
        vru_state = VRUState.from_model(vru)

        ctrl_out = ego_ctrl.plan(
            veh0=veh_state,
            vru0=vru_state,
            delta0=delta_cmd,
            ref_speed=ref_speed,
            detected=bool(detected),
            belief=belief,
            scenario=SCENARIO_TAG,
            vru_react_when_undetected=True,
        )

        a_cmd = float(ctrl_out.a_cmd)
        delta_cmd = float(ctrl_out.delta_cmd)

        # use true rider type for simulation rollout
        a0_by_type = (ctrl_out.debug or {}).get("a0_by_type", None)
        if isinstance(a0_by_type, dict) and (TRUE_RIDER in a0_by_type):
            a0 = a0_by_type[TRUE_RIDER]
            vru_mode = str(a0.mode) if hasattr(a0, "mode") else str(a0)
        else:
            vru_mode = str((ctrl_out.debug or {}).get("vru_action_0", "go"))

        # 4) apply dynamics
        veh.update_state(a=a_cmd, delta=delta_cmd)

        target_xy, v_des = path_mgr.step_target(vru, vru_mode)
        Fx, Fy = sfm_force_to_target(vru, target_xy=target_xy, v_des=v_des, tau=0.5)
        vru.update(F_x=Fx, F_y=Fy)

        # 5) collision + evaluation
        col = check_collision(veh, vru)

        metric.cal_eff_index(actual_finish_time=float(veh.t), ref_x=ref_x, ref_y=ref_y)
        metric.cal_ttc_ind(collision_state=bool(col))
        metric.cal_safety_index(v_impact=float(veh.v) if col else None)

        metric.jerk_traj = []
        metric.comfort_index_traj = []
        metric.cal_cmft_index(acc_traj=veh.acc_traj,yaw_hist=veh.yaw_hist)
        metric.cal_overall_index(coe_eff=1.0, coe_safety=1.0, coe_cmft=1.0)

        # 6) print key states
        if step % 5 == 0:
            s_dbg = belief_dbg.get("go_score_s", float("nan"))
            pgo_dbg = belief_dbg.get("p_go_obs", float("nan"))
            print(
                f"[t={veh.t:5.2f}] ctrl={ego_controller:5s} "
                f"veh(x={veh.x:6.2f}, v={veh.v:5.2f}, a={a_cmd:5.2f}, d={delta_cmd:5.3f}) | "
                f"vru(x={vru.x:6.2f}, y={vru.y:6.2f}, spd={math.hypot(vru.v_x,vru.v_y):4.2f}, mode={vru_mode:5s}) | "
                f"FOV={fov_flag} V2X={v2x_detected} det={int(detected)} | "
                f"belief P(aggr)={belief.p_aggr:4.2f} (s={s_dbg:4.2f}, p_go={pgo_dbg:4.2f}) | "
                f"collision={int(col)}"
            )

        # 7) log
        log["t"].append(float(veh.t))
        log["veh_x"].append(float(veh.x))
        log["veh_v"].append(float(veh.v))
        log["veh_a"].append(float(a_cmd))
        log["veh_delta"].append(float(delta_cmd))

        log["vru_x"].append(float(vru.x))
        log["vru_y"].append(float(vru.y))
        log["vru_v"].append(float(math.hypot(vru.v_x, vru.v_y)))

        log["fov"].append(int(fov_flag))
        log["v2x"].append(int(v2x_detected))
        log["detected"].append(int(detected))

        log["vru_mode"].append(str(vru_mode))
        log["collision"].append(int(col))

        log["p_aggr"].append(float(belief.p_aggr))
        log["go_score_s"].append(float(belief_dbg.get("go_score_s", float("nan"))))
        log["p_go_obs"].append(float(belief_dbg.get("p_go_obs", float("nan"))))

        log["safety"].append(float(metric.safety_index))
        log["eff"].append(float(metric.eff_index))
        log["comfort"].append(float(metric.comfort_index))
        log["overall"].append(float(metric.overall_index))

        cur_bin = None
        if hasattr(metric, "ttc_ind_traj") and metric.ttc_ind_traj:
            cur_bin = int(metric.ttc_ind_traj[-1])
        ttc_bin_hist.append(cur_bin)

        # 8) render + save frame
        if render and renderer is not None:
            try:
                renderer.update(
                    frame_idx=step,
                    veh=veh,
                    vru=vru,
                    veh_sensor=veh_sensor,
                    metric=metric,
                    v2x_profile=v2x_profile,
                    rider_type=rider_type,
                    a_cmd=a_cmd,
                    delta_cmd= delta_cmd,
                    vru_mode=vru_mode,
                    fov_flag=fov_flag,
                    v2x_flag=v2x_detected,
                    detected=int(detected),
                    ttc_bin_hist=ttc_bin_hist,
                    controller=str(ego_controller),
                )
            except TypeError:
                # backward-compatible renderer signature
                renderer.update(
                    frame_idx=step,
                    veh=veh,
                    vru=vru,
                    veh_sensor=veh_sensor,
                    metric=metric,
                    v2x_profile=v2x_profile,
                    rider_type=rider_type,
                    a_cmd=a_cmd,
                    delta_cmd=delta_cmd,
                    vru_mode=vru_mode,
                    fov_flag=fov_flag,
                    v2x_flag=v2x_detected,
                    detected=int(detected),
                    ttc_bin_hist=ttc_bin_hist,
                )

        # 9) break conditions
        if col:
            print(f"*** COLLISION @ t={veh.t:.2f}s ***")
            break
        if float(veh.x) >= 30.0:
            print(f"*** VEHICLE FINISHED @ t={veh.t:.2f}s ***")
            break

    if render and renderer is not None:
        renderer.close()
        plt.ioff()
        plt.show()

    return log


def img2video(out_dir: str,
              fps: int = 10,
              out_name: str = "simulation.mp4",
              scale: str = "1280:1024",
              pattern: str = "Frame%04d.png"):
    out_dir = os.path.normpath(out_dir)
    in_pattern = os.path.join(out_dir, pattern)
    out_path = os.path.join(out_dir, out_name)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", in_pattern,
        "-vf", f"scale={scale}",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[INFO] Video saved to: {out_path}")

def replay_collisions_from_xlsx(
    xlsx_path: str,
    ctrl_filter: list | None = None,
    max_steps: int = 300,
    render: bool = True,
    save_frames: bool = True,
    base_out_dir: str = os.path.join("qual_sim", "replay_output", "straight_road"),
    make_video: bool = True,
):
    """Replay collision cases from a quant straight-road collision xlsx file.

    Parameters
    ----------
    xlsx_path   : path to the collisions_*.xlsx produced by test_quant_straight_road.py
    ctrl_filter : if None/[], replay all controllers; otherwise only those in the list
    max_steps   : episode length (match quant default of 300)
    base_out_dir: parent folder; replay subfolder is named replay_<xlsx_stem>
    make_video  : run ffmpeg on each case folder after rendering
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for replay: pip install openpyxl")

    xlsx_path = os.path.normpath(xlsx_path)
    xlsx_stem = os.path.splitext(os.path.basename(xlsx_path))[0]
    replay_dir = os.path.join(base_out_dir, f"replay_{xlsx_stem}")
    os.makedirs(replay_dir, exist_ok=True)
    print(f"[REPLAY] Output directory: {replay_dir}")

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        print("[REPLAY] Empty xlsx, nothing to do.")
        return

    header = [str(c) if c is not None else "" for c in rows[0]]
    col = {name: idx for idx, name in enumerate(header)}

    def _get(row, name, default=None):
        idx = col.get(name)
        if idx is None:
            return default
        v = row[idx]
        return default if v is None else v

    data_rows = rows[1:]
    print(f"[REPLAY] {len(data_rows)} collision rows found in {xlsx_stem}")

    for i, row in enumerate(data_rows):
        ctrl = str(_get(row, "controller", "mpc1d"))
        rider = str(_get(row, "rider_type", "Conservative"))
        profile = str(_get(row, "v2x_profile", "no"))
        seed = int(_get(row, "seed", 42))
        run_id = _get(row, "run_id", i)
        cfg_id = _get(row, "base_cfg_id", 0)
        speed = float(_get(row, "veh_speed", 14.58))
        veh_y0 = float(_get(row, "veh_y0", -2.0))
        vru_x0 = float(_get(row, "vru_x0", -30.0))
        vru_y0 = float(_get(row, "vru_y0", 6.0))
        y_start = float(_get(row, "y_start", 6.0))
        y_target = float(_get(row, "y_target", -6.0))
        x_lc_start = float(_get(row, "x_lc_start", -15.0))
        x_lc_done = float(_get(row, "x_lc_done", 0.0))
        lookahead = float(_get(row, "lookahead", 12.0))
        y_eps = float(_get(row, "y_eps", 0.6))

        if ctrl_filter and ctrl not in ctrl_filter:
            continue

        folder_name = (
            f"case{i:03d}_run{run_id}_cfg{cfg_id}_{ctrl}_{rider}_{profile}"
            f"_v{speed:.1f}_s{seed}"
        )
        case_dir = os.path.join(replay_dir, folder_name)
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[REPLAY {i}] {folder_name}")
        run_simulation(
            v2x_profile=profile,
            rider_type=rider,
            ego_controller=ctrl,
            max_steps=max_steps,
            render=render,
            save_frames=save_frames,
            out_dir=case_dir,
            seed=seed,
            veh_speed=speed,
            veh_y0=veh_y0,
            vru_x0=vru_x0,
            vru_y0=vru_y0,
            y_start=y_start,
            y_target=y_target,
            x_lc_start=x_lc_start,
            x_lc_done=x_lc_done,
            lookahead=lookahead,
            y_eps=y_eps,
        )

        if make_video and save_frames:
            try:
                img2video(case_dir)
            except Exception as e:
                print(f"[REPLAY] Video generation failed for {folder_name}: {e}")

    print(f"\n[REPLAY] Done. Results in: {replay_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="no", choices=["no", "bad", "good", "perfect"])
    parser.add_argument("--rider", type=str, default="Aggressive", choices=["Aggressive", "Conservative"])
    parser.add_argument("--ego", type=str, default="mpc2d",
                        help="Controller name(s), comma-separated (e.g. mpc1d,mpc2d). "
                             "For single run: one name. For replay: filter list (default: all).")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", type=str, default=None,
                        help="Path to collision xlsx from quant_straight_road. "
                             "Triggers replay mode (ignores --profile/--rider/--ego for selection).")
    parser.add_argument("--replay-out", type=str,
                        default=os.path.join("qual_sim", "replay_output", "straight_road"),
                        help="Base output directory for replay folders.")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip ffmpeg video generation after replay.")
    args = parser.parse_args()

    render = True
    if args.no_render:
        render = False
    elif args.render:
        render = True

    if args.replay:
        ctrl_filter = [c.strip() for c in args.ego.split(",") if c.strip()] if args.ego else None
        replay_collisions_from_xlsx(
            xlsx_path=args.replay,
            ctrl_filter=ctrl_filter if ctrl_filter else None,
            max_steps=args.steps,
            render=render,
            save_frames=(not args.no_save),
            base_out_dir=args.replay_out,
            make_video=(not args.no_video),
        )
        return

    tic = time.perf_counter()
    out_dir = os.path.join("qual_sim", "output", "straight_road")
    run_simulation(
        v2x_profile=args.profile,
        rider_type=args.rider,
        ego_controller=args.ego,
        max_steps=args.steps,
        render=render,
        save_frames=(not args.no_save),
        seed=args.seed,
        out_dir=out_dir,
    )

    img2video(out_dir)

    toc = time.perf_counter()
    print(f"Simulation finished in {toc - tic:.3f}s")


if __name__ == "__main__":
    main()
