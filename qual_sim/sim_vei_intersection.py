# qual_sim/sim_vei_intersection.py
from __future__ import annotations

import os
import subprocess
import math
import time
import argparse
import numpy as np

from qual_sim.render_vei_intersection import VEIIntersectionRenderer

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

# for mpc2d debug
from interaction.controllers.mpc_2d import MPC2DConfig

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

# ----------------------- scenario VRU path manager -----------------------
class VRUPathManager:
    """
    Two-stage path:
      stage0: go to turn point (intersection center)
      stage1: go to destination (left turn to west)

    yield mode: go to stop point (or hold) with v_des=0
    """
    def __init__(self, stop_pt, turn_pt, dest_pt, arrive_eps=1.2,
                 v_go: float = 10.0, v_yield: float = 7.0,):

        self.stop_pt = tuple(stop_pt)
        self.turn_pt = tuple(turn_pt)
        self.dest_pt = tuple(dest_pt)
        self.arrive_eps = float(arrive_eps)
        self.stage = 0

        self.v_go = float(v_go)
        self.v_yield = float(v_yield)

    def _near(self, vru, pt):
        return math.hypot(float(vru.x)-pt[0], float(vru.y)-pt[1]) <= self.arrive_eps

    def step_target(self, vru, vru_mode: str):
        # stage update only in "go"
        if vru_mode == "go" and self.stage == 0 and self._near(vru, self.turn_pt):
            self.stage = 1

        if vru_mode == "yield":
            stop_y = float(self.stop_pt[1])

            # if already at/above stop_y, stop in place (do NOT chase stop_pt backwards)
            if float(vru.y) >= stop_y - 0.2:
                return (float(vru.x), float(vru.y)), 0.0

            # does not reach stop line yet
            return self.stop_pt, 0.0

        else:
            if self.stage == 0:
                return self.turn_pt, 5.0
            else:
                return self.dest_pt, 5.0


def run_simulation(
    v2x_profile: str = "no",
    rider_type: str = "Conservative",
    ego_controller: str = "mpc1d",
    max_steps: int = 300,
    render: bool = True,
    save_frames: bool = True,
    out_dir: str | None = None,
    seed: int = 42,
    # --- per-episode parameters (match quant_sim EpisodeConfig) ---
    veh_speed: float = 15.73,
    vru_x0: float = 4.0,
    vru_y0: float = -20.0,
    stop_x: float = 4.0,
    stop_y: float = -14.0,
    turn_x: float = 4.0,
    turn_y: float = 0.0,
    dest_x: float = -18.0,
    dest_y: float = 20.0,
):
    # ------------------- output dir -------------------
    if out_dir is None:
        out_dir = os.path.join("qual_sim", "output", "intersection")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------- layout -------------------
    layout = TrafficLayout(
        "vei_intersection",
        X_min=-35, X_max=35,
        Y_min=-35, Y_max=35,
        W_lane=4.0
    )

    # ------------------- vehicle init (straight eastbound along +x) -------------------
    veh = BicycleModel(
        params=BicycleModel_params,
        x=-70.0, y=-6.0,
        psi=0.0, v=float(veh_speed),
        t0=0.0, beta=0.0, yaw=0.0
    )
    delta_cmd = 0.0  # longitudinal only
    ref_speed = float(veh_speed)

    # reference path for efficiency metric (straight line in x)
    ref_x = np.linspace(float(veh.x), 30.0, 300)
    ref_y = np.zeros_like(ref_x)

    # ------------------- VRU init (escooter from south -> center -> left to west) -------------------
    vru = PointMassNewton(
        params=PointMass_params["escooter"],
        position=[float(vru_x0), float(vru_y0)],
        velocity=[0.0, 0.0],
        dt=veh.dt,
        vru_category="escooter",
        rider_type=rider_type
    )


    path_mgr = VRUPathManager(
        stop_pt=(float(stop_x), float(stop_y)),
        turn_pt=(float(turn_x), float(turn_y)),
        dest_pt=(float(dest_x), float(dest_y)),
        arrive_eps=1.2
    )
    #----belief in stackelberg----
    belief = RiderTypeBelief(p_aggr=0.5, alpha=1.0)  # vehicle prior: unknown type
    TRUE_RIDER = rider_type  # this is the true VRU type in the simulation
    SCENARIO_TAG = "intersection"

    # ------------------- perception -------------------
    # Keep simple. Make FOV shorter so V2X can matter.
    veh_sensor = VehFOV(
        fov_range=30.0,
        fov_angle=2*math.pi/3,
        vehicle=veh,
        vru=vru
    )

    vru_awr = VRU_AWR(profile=v2x_profile, init_state=1, seed=seed)
    v2x_range = 40.0

    # ------------------- game MPC -------------------
    AV = [VehicleAction(a) for a in (-5.0,-4.0,-3.0,-2.0,-1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    AS = [VRUAction("yield"), VRUAction("go")]

    wv = PayoffWeights(
        w_collision_v=1e6,
        w_ttc_v=8e3,
        w_speed_v=6.0,
        w_acc_v=0.4,
        w_jerk_v=0.1
    )
    # ------------------- interaction oracle + swappable ego controller (Step 5) -------------------
    oracle = InteractionOracle(horizon_steps=20, vru_actions=AS)

    cfg = MPC2DConfig(log_dir=os.path.join("qual_sim", "output", "intersection", "logs"))

    ego_ctrl = make_ego_controller(
        ego_controller,
        oracle=oracle,
        vehicle_actions=AV,
        weights_vehicle=wv,
        verbose=True,
    )

    # ------------------- evaluator -------------------
    metric = SustainEvaluator(
        start_pos=float(veh.x),
        finish_pos=30.0,
        veh=veh,
        desired_veh_velo=ref_speed,
        vru=vru
    )

    # ------------------- logs -------------------
    log = {
        "t": [], "veh_x": [], "veh_v": [], "veh_a": [],
        "vru_x": [], "vru_y": [], "vru_v": [],
        "fov": [], "v2x": [], "detected": [],
        "vru_mode": [], "collision": [],
        "safety": [], "eff": [], "comfort": [], "overall": [],
        # belief debug (leader belief about rider aggressiveness)
        "p_aggr": [], "go_score_s": [], "p_go_obs": []
    }

    # store per-step TTC-bin for tail coloring (Option 1)
    ttc_bin_hist: list[int | None] = []

    renderer = None
    if render:
        renderer = VEIIntersectionRenderer(
            layout=layout,
            veh_sensor=veh_sensor,
            out_dir=out_dir,
            save_frames=save_frames,
            xlim=(-30, 30),
            ylim=(-30, 30),
            tail_len=30,  # adjust here
            show_full_traj=True,  # optional
        )


    # ------------------- anti-deadlock params (SYNC with quant) -------------------
    YIELD_FORCE_GO_STEPS = 60
    VEH_CREEP_AFTER_STEPS = 40
    V_STOP_EPS = 0.20
    X_STUCK_EPS = 0.05
    SAFE_DIST_FOR_CREEP = 6.0
    CREEP_ACCEL = 1.0
    DEADLOCK_STEPS = 100

    # Rule-3 post-dynamics deadlock tracker
    X_PROGRESS_EPS = 0.01
    deadlock_streak = 0

    # override latches
    FORCE_GO_HOLD_STEPS = 25
    CREEP_HOLD_STEPS = 20
    YIELD_LOCK_STEPS = 30

    STOP_EPS = 1.2
    VRU_STOP_SPD = 0.25

    vru_force_go_timer = 0
    veh_creep_timer = 0
    vru_yield_lock_timer = 0

    consecutive_yield = 0
    forced_vru_go = 0
    forced_veh_creep = 0

    veh_x_prev = float(veh.x)
    vru_x_prev = float(vru.x)
    vru_y_prev = float(vru.y)

    # ------------------- sim loop -------------------
    for step in range(int(max_steps)):
        # 1) perception update (called exactly once!)
        fov_flag = int(veh_sensor.update())

        link_state = int(vru_awr.link_check())  # 0/1
        dist_vru = math.hypot(float(vru.x)-float(veh.x), float(vru.y)-float(veh.y))
        v2x_detected = 1 if (link_state == 1 and dist_vru <= v2x_range) else 0

        detected_fov = (fov_flag == 1)
        detected = detected_fov  # FOV only — matches quant (V2X excluded from planner signal)

        # --- threat gating (SYNC with quant) ---
        ttc_now = ttc_los(veh, vru)
        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))

        D_TH = 40.0
        TTC_TH = 6.0
        threat = (dist_vru <= D_TH) and (ttc_now < TTC_TH)
        detected_for_planner = bool(detected and threat)

        # Belief update is driven ONLY by motion cues that require *visual* observation.
        # So we gate by FOV detection (not V2X), making it scenario-agnostic.
        belief_dbg: dict = {}
        belief.update_from_motion(
            detected=bool(fov_flag == 1),
            scenario=SCENARIO_TAG,
            vru_x=float(vru.x), vru_y=float(vru.y),
            vru_vx=float(vru.v_x), vru_vy=float(vru.v_y),
            debug=belief_dbg,
        )

        # 2) game MPC decide (veh accel + vru mode)
        DEBUG_MPC2D = False
        DEBUG_EVERY = 1  # start with every step, then increase later


        veh_state = VehState.from_model(veh)
        vru_state = VRUState.from_model(vru)
        ctrl_out = ego_ctrl.plan(
            veh0=veh_state,
            vru0=vru_state,
            delta0=delta_cmd,
            ref_speed=ref_speed,
            detected=detected_for_planner,
            belief=belief,
            scenario=SCENARIO_TAG,
            # debug_costs=True,  #turn on candidate breakdown
            debug_costs= DEBUG_MPC2D,
            vru_react_when_undetected=True,
        )

        a_cmd = float(ctrl_out.a_cmd)
        delta_cmd = float(ctrl_out.delta_cmd)

        #====== mpc2d debug ======
        dbg = ctrl_out.debug or {}
        if DEBUG_MPC2D:
            print(f"\n [t={veh.t:.2f}]  "
                  f"\n contract={dbg.get('contract')}"
                  f"\n overide={dbg.get('override')}"
                  f"\n strat={dbg.get('strat')}"
                  )

        if DEBUG_MPC2D and (step % DEBUG_EVERY == 0):
            print(f"  chosen: a={ctrl_out.a_cmd:+.2f}, delta={ctrl_out.delta_cmd:+.3f} rad "
                  f"ttc_min_b={dbg.get('ttc_min_b')} trigger={dbg.get('ttc_trigger')}")

        # beam = dbg.get("beam_log", None)
        # if DEBUG_MPC2D and beam and (step % DEBUG_EVERY == 0):
        #     print("  beam top:")
        #     for row in beam[:3]:
        #         print("   ", row)

        # --- IMPORTANT: use true rider type to select VRU action ---
        a0_by_type = (ctrl_out.debug or {}).get("a0_by_type", None)

        if isinstance(a0_by_type, dict) and (TRUE_RIDER in a0_by_type):
            # if stored as VRUAction objects:
            if hasattr(a0_by_type[TRUE_RIDER], "mode"):
                vru_mode = str(a0_by_type[TRUE_RIDER].mode)
            else:
                # if stored as strings
                vru_mode = str(a0_by_type[TRUE_RIDER])
        else:
            # fallback to whatever plan() returned (legacy behavior)
            vru_mode = str((ctrl_out.debug or {}).get("vru_action_0", "go"))

        # =================== Yield lock until AV passes conflict (SYNC with quant) ===================
        X_CONFLICT = float(path_mgr.stop_pt[0])  # stop_x == turn_x in this scenario
        PASS_X_MARGIN = 3.0

        vru_speed = math.hypot(float(vru.v_x), float(vru.v_y))
        near_stop = (math.hypot(float(vru.x) - float(path_mgr.stop_pt[0]),
                                float(vru.y) - float(path_mgr.stop_pt[1])) <= STOP_EPS)
        vru_stopped_at_stop = bool(near_stop and (vru_speed <= VRU_STOP_SPD))

        veh_passed_conflict = bool(float(veh.x) >= X_CONFLICT + PASS_X_MARGIN)

        if vru_stopped_at_stop and (not veh_passed_conflict):
            vru_mode = "yield"
        # ================================================================================

        # --- Late-yield guard (SYNC with quant) ---
        if vru_mode == "yield" and float(vru.y) > float(path_mgr.stop_pt[1]) + 0.5:
            vru_mode = "go"

        # ------------------- apply latches (SYNC with quant) -------------------
        if vru_force_go_timer > 0:
            vru_mode = "go"
            vru_force_go_timer -= 1

        if vru_yield_lock_timer > 0 and vru_force_go_timer <= 0:
            vru_mode = "yield"
            vru_yield_lock_timer -= 1

        if veh_creep_timer > 0:
            a_cmd = max(float(a_cmd), CREEP_ACCEL)
            veh_creep_timer -= 1

        # ------------------- anti-deadlock Rule1/Rule2 (SYNC with quant) -------------------
        if vru_mode == "yield":
            consecutive_yield += 1
        else:
            consecutive_yield = 0

        veh_stopped = abs(float(veh.v)) < V_STOP_EPS
        veh_no_progress = abs(float(veh.x) - veh_x_prev) < X_STUCK_EPS

        vru_no_progress = (
                abs(float(vru.x) - vru_x_prev) < X_STUCK_EPS
                and abs(float(vru.y) - vru_y_prev) < X_STUCK_EPS
        )

        dist_vru = math.hypot(float(vru.x) - float(veh.x), float(vru.y) - float(veh.y))

        # Rule 1
        if vru_mode == "yield" and consecutive_yield >= YIELD_FORCE_GO_STEPS:
            if vru_stopped_at_stop:
                veh_creep_timer = max(veh_creep_timer, CREEP_HOLD_STEPS)
                vru_yield_lock_timer = max(vru_yield_lock_timer, YIELD_LOCK_STEPS)
                forced_veh_creep += 1
            else:
                vru_force_go_timer = max(vru_force_go_timer, FORCE_GO_HOLD_STEPS)
                forced_vru_go += 1
            consecutive_yield = 0

        # Rule 2
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

        # Store the previous states for anti-deadlock rule 3
        veh_x_old = float(veh.x)
        vru_x_old = float(vru.x)
        vru_y_old = float(vru.y)

        # 3) apply real dynamics
        veh.update_state(a=a_cmd, delta=delta_cmd)

        target_xy, v_des = path_mgr.step_target(vru, vru_mode)
        Fx, Fy = sfm_force_to_target(vru, target_xy=target_xy, v_des=v_des, tau=0.6)
        vru.update(F_x=Fx, F_y=Fy)

        # ------------------- Rule-3: post-dynamics progress check (SYNC with quant) -------------------
        veh_moved = abs(float(veh.x) - veh_x_old) >= X_PROGRESS_EPS
        vru_moved = (
                abs(float(vru.x) - vru_x_old) >= X_PROGRESS_EPS
                or abs(float(vru.y) - vru_y_old) >= X_PROGRESS_EPS
        )

        if veh_moved or vru_moved:
            deadlock_streak = 0
        else:
            deadlock_streak += 1

        if deadlock_streak >= DEADLOCK_STEPS:
            if vru_stopped_at_stop:
                veh_creep_timer = max(veh_creep_timer, CREEP_HOLD_STEPS * 2)
                vru_yield_lock_timer = max(vru_yield_lock_timer, YIELD_LOCK_STEPS * 2)
                forced_veh_creep += 1
                deadlock_streak = 0
            else:
                print(f"*** DEADLOCK @ t={veh.t:.2f}s ***")
                break

        # update prev-position trackers (SYNC with quant): enables Rule 2 per-step progress check
        veh_x_prev = float(veh.x)
        vru_x_prev = float(vru.x)
        vru_y_prev = float(vru.y)

        # 4) collision + evaluation (match your SustainEvaluator API!)
        col = check_collision(veh, vru)

        metric.cal_eff_index(actual_finish_time=float(veh.t), ref_x=ref_x, ref_y=ref_y)
        # cal_ttc_ind called per-step (post-dynamics) — collision state is always current;
        # has_collision is set True at the collision step before break, matching quant end-mode result.
        metric.cal_ttc_ind(collision_state=bool(col))
        metric.cal_safety_index(v_impact=float(veh.v) if col else None)

        # avoid repeated accumulation in comfort
        metric.jerk_traj = []
        metric.comfort_index_traj = []
        metric.cal_cmft_index(acc_traj=veh.acc_traj,yaw_hist=veh.yaw_hist)

        metric.cal_overall_index(coe_eff=1.0, coe_safety=1.0, coe_cmft=1.0)

        # 5) print key states
        if step % 5 == 0:
            s_dbg = belief_dbg.get("go_score_s", float("nan"))
            pgo_dbg = belief_dbg.get("p_go_obs", float("nan"))
            print(
                f"[t={veh.t:5.2f}] "
                f"veh(x={veh.x:6.2f}, v={veh.v:5.2f}, a={a_cmd:5.2f}) | "
                f"vru(x={vru.x:6.2f}, y={vru.y:6.2f}, spd={math.hypot(vru.v_x,vru.v_y):4.2f}, mode={vru_mode:5s}) | "
                f"FOV={fov_flag} V2X={v2x_detected} det={int(detected)} | "
                f"belief P(aggr)={belief.p_aggr:4.2f} (s={s_dbg:4.2f}, p_go={pgo_dbg:4.2f}) | "
                
                f"safety score={metric.safety_index:5.2f}) | "
                # f"score(sfty={metric.safety_index:5.2f}, eff={metric.eff_index:5.2f}, cmft={metric.comfort_index:5.2f}, all={metric.overall_index:5.2f}) | "
                f"collision={int(col)}"
            )

        # ================= debug block ============
        dbg = ctrl_out.debug or {}
        cands = dbg.get("candidates", [])
        # if cands:
        if cands and step % 5 == 0:
            # sort by expected cost (lower is better)
            cands_sorted = sorted(cands, key=lambda r: r.get("Jv_exp", 1e18))
            print(f"[{ego_controller}] belief={dbg.get('belief')}")
            for r in cands_sorted[:3]:
                print(f"  a={r['a']:+.2f}  Jexp={r['Jv_exp']:.2f}  "
                      f"J_A={r['Jv_by_type'].get('Aggressive'):.2f}  J_C={r['Jv_by_type'].get('Conservative'):.2f}  "
                      f"ttcA={r['min_ttc_by_type'].get('Aggressive'):.2f}  ttcC={r['min_ttc_by_type'].get('Conservative'):.2f}")
        # ================= debug block ============

        # 6) log
        log["t"].append(float(veh.t))
        log["veh_x"].append(float(veh.x))
        log["veh_v"].append(float(veh.v))
        log["veh_a"].append(float(a_cmd))
        log["vru_x"].append(float(vru.x))
        log["vru_y"].append(float(vru.y))
        log["vru_v"].append(float(math.hypot(vru.v_x, vru.v_y)))
        log["fov"].append(int(fov_flag))
        log["v2x"].append(int(v2x_detected))
        log["detected"].append(int(detected))
        log["vru_mode"].append(vru_mode)
        log["collision"].append(int(col))
        log["safety"].append(float(metric.safety_index))
        log["eff"].append(float(metric.eff_index))
        log["comfort"].append(float(metric.comfort_index))
        log["overall"].append(float(metric.overall_index))

        # belief logs (safe even when not detected)
        log["p_aggr"].append(float(belief.p_aggr))
        log["go_score_s"].append(float(belief_dbg.get("go_score_s", float("nan"))))
        log["p_go_obs"].append(float(belief_dbg.get("p_go_obs", float("nan"))))

        # 7) render + save frame
        cur_bin = None
        if hasattr(metric, "ttc_ind_traj") and metric.ttc_ind_traj:
            cur_bin = int(metric.ttc_ind_traj[-1])
        ttc_bin_hist.append(cur_bin)

        if render and renderer is not None:
            renderer.draw_frame(
                step=step,
                veh=veh,
                vru=vru,
                vru_mode=vru_mode,
                fov_flag=fov_flag,
                v2x_detected=v2x_detected,
                detected=int(detected),
                metric=metric,
                a_cmd=a_cmd,
                delta_cmd=delta_cmd,
                v2x_profile=v2x_profile,
                rider_type=rider_type,
                ttc_bin_hist=ttc_bin_hist,
                controller=str(ego_controller),
            )

        # 8) break conditions
        if col:
            print(f"*** COLLISION @ t={veh.t:.2f}s ***")
            break
        if veh.x >= 30.0:
            print(f"*** VEHICLE FINISHED @ t={veh.t:.2f}s ***")
            break

    if render and renderer is not None:
        renderer.close()

    return log


def replay_collisions_from_xlsx(
    xlsx_path: str,
    ctrl_filter: list | None = None,
    max_steps: int = 400,
    render: bool = True,
    save_frames: bool = True,
    base_out_dir: str = os.path.join("qual_sim", "replay_output", "intersection"),
    make_video: bool = True,
):
    """
    Load collision rows from a quant_sim collisions xlsx and replay each in qual_sim.

    ctrl_filter: list of controller names to replay (e.g. ["mpc1d", "mpc2d"]).
                 If None or empty, all controllers in the xlsx are replayed.

    The replay output folder is named  replay_<xlsx_stem>  under base_out_dir,
    e.g.  replay_quant_vei_intersection_collisions_20260401_140633

    Expected xlsx columns (from quant_sim collision_header):
      run_id, cfg_id, controller, v2x_profile, rider_type,
      veh_x0, veh_y0, veh_yaw0, veh_speed, seed,
      vru_x0, vru_y0, stop_x, stop_y, turn_x, turn_y, dest_x, dest_y,
      end_reason, steps, t_end, min_ttc
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required to read xlsx files: pip install openpyxl")

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        print("[REPLAY] xlsx is empty.")
        return

    header = [str(h) for h in rows[0]]
    data_rows = rows[1:]

    # controller filter: None / [] means "all"
    active_filter = [c.strip() for c in ctrl_filter] if ctrl_filter else []

    # derive replay folder name from the xlsx filename
    xlsx_stem = os.path.splitext(os.path.basename(xlsx_path))[0]
    replay_dir = os.path.join(base_out_dir, f"replay_{xlsx_stem}")
    os.makedirs(replay_dir, exist_ok=True)

    # apply filter for reporting
    if active_filter:
        selected = [r for r in data_rows if str(r[header.index("controller")]) in active_filter]
        print(f"[REPLAY] {len(selected)}/{len(data_rows)} collision case(s) match "
              f"filter {active_filter} — from {xlsx_path}")
    else:
        selected = list(data_rows)
        print(f"[REPLAY] {len(selected)} collision case(s) (all controllers) — from {xlsx_path}")

    def col(row, name):
        return row[header.index(name)]

    for i, row in enumerate(selected):
        run_id  = col(row, "run_id")
        cfg_id  = col(row, "cfg_id")
        ctrl    = str(col(row, "controller"))
        profile = str(col(row, "v2x_profile"))
        rider   = str(col(row, "rider_type"))
        speed   = float(col(row, "veh_speed"))
        seed    = int(col(row, "seed"))
        vru_x0  = float(col(row, "vru_x0"))
        vru_y0  = float(col(row, "vru_y0"))
        stop_x  = float(col(row, "stop_x"))
        stop_y  = float(col(row, "stop_y"))
        turn_x  = float(col(row, "turn_x"))
        turn_y  = float(col(row, "turn_y"))
        dest_x  = float(col(row, "dest_x"))
        dest_y  = float(col(row, "dest_y"))

        case_dir = os.path.join(
            replay_dir,
            f"case{i:03d}_run{run_id}_cfg{cfg_id}_{ctrl}_{rider}_{profile}_v{speed:.1f}_s{seed}"
        )
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[REPLAY {i+1}/{len(selected)}] "
              f"run_id={run_id} cfg_id={cfg_id} ctrl={ctrl} "
              f"rider={rider} profile={profile} speed={speed:.1f} seed={seed} "
              f"vru0=({vru_x0},{vru_y0})")

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
            vru_x0=vru_x0,
            vru_y0=vru_y0,
            stop_x=stop_x,
            stop_y=stop_y,
            turn_x=turn_x,
            turn_y=turn_y,
            dest_x=dest_x,
            dest_y=dest_y,
        )

        if make_video and save_frames:
            img2video(case_dir, out_name=f"case{i:03d}.mp4")


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="no", choices=["no", "bad", "good", "perfect"])
    parser.add_argument("--rider", type=str, default="Aggressive", choices=["Aggressive", "Conservative"])
    parser.add_argument("--ego", type=str, default=None,
                        help="Single-run: one of rule1d/rule2d/mpc1d/mpc2d. "
                             "Replay mode: comma-separated list to filter, e.g. mpc1d,mpc2d. "
                             "Omit (or leave blank) in replay mode to reproduce all controllers.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    # --- collision replay ---
    parser.add_argument("--replay", type=str, default=None,
                        metavar="COLLISIONS_XLSX",
                        help="Path to quant_sim collisions xlsx. Replays each collision case in qual_sim.")
    parser.add_argument("--replay-out", type=str,
                        default=os.path.join("qual_sim", "replay_output", "intersection"),
                        help="Base directory under which replay_<xlsx_stem> folder is created.")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip ffmpeg video generation after replay.")
    args = parser.parse_args()

    tic = time.perf_counter()

    if args.replay:
        ctrl_filter = [c.strip() for c in args.ego.split(",")] if args.ego else None
        replay_collisions_from_xlsx(
            xlsx_path=args.replay,
            ctrl_filter=ctrl_filter,
            max_steps=args.steps,
            render=True,
            save_frames=(not args.no_save),
            base_out_dir=args.replay_out,
            make_video=(not args.no_video),
        )
    else:
        ego = args.ego if args.ego else "mpc2d"
        out_dir = os.path.join("qual_sim", "output", "intersection")
        run_simulation(
            v2x_profile=args.profile,
            rider_type=args.rider,
            ego_controller=ego,
            max_steps=args.steps,
            render=True,
            save_frames=(not args.no_save),
            seed=args.seed
        )
        img2video(out_dir)

    toc = time.perf_counter()
    print(f"Simulation finished in {toc - tic:.3f}s")


if __name__ == "__main__":
    main()
