from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from .state import VehState, VRUState
from .dynamics import VehicleAction, VRUAction


@dataclass
class PayoffWeights:
    # vehicle
    w_collision_v: float = 1e6
    w_ttc_v: float = 2e3
    w_speed_v: float = 5.0
    w_acc_v: float = 0.5
    w_jerk_v: float = 0.2

    # vru
    w_collision_s: float = 1e6
    w_ttc_s: float = 0.0  # <-- NEW: VRU TTC risk weight
    w_progress_s: float = 5.0
    w_effort_s: float = 0.2


def vehicle_radius(veh: VehState) -> float:
    # conservative bounding circle
    return 0.5 * math.hypot(veh.params.LENGTH, veh.params.WIDTH)


def collision(veh: VehState, vru: VRUState) -> bool:  # TODO: May need to  consider more precise collision check
    r = vehicle_radius(veh) + vru.params.R
    d = math.hypot(vru.x - veh.x, vru.y - veh.y)
    return d <= r


def ttc(veh: VehState, vru: VRUState) -> float:
    """
    Simple line-of-sight TTC.
    If not closing, TTC = +inf.
    """
    rx = vru.x - veh.x
    ry = vru.y - veh.y
    dist = math.hypot(rx, ry)
    if dist < 1e-6:
        return 0.0

    # relative velocity projected on line-of-sight
    # approximate vehicle velocity vector by heading yaw
    vvx = veh.v * math.cos(veh.yaw)
    vvy = veh.v * math.sin(veh.yaw)

    rvx = vru.vx - vvx
    rvy = vru.vy - vvy

    closing = -(rx * rvx + ry * rvy) / dist  # >0 means closing
    if closing <= 1e-6:
        return float("inf")
    return dist / closing


def stage_costs(
    veh: VehState,
    vru: VRUState,
    a_v: VehicleAction,
    a_s: VRUAction,
    ref_speed: float,
    u_last: float,
    w: PayoffWeights,
    w_vru: Optional[PayoffWeights] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Return (J_vehicle, J_vru, debug_terms).

    w       — weight set for vehicle (leader) cost terms.
    w_vru   — weight set for VRU (follower) cost terms.  If None, `w` is used
              for both (backward-compatible single-weight call).

    Passing both weight sets allows the oracle to compute Jv and Js in a
    single call, halving the number of geometry evaluations (collision + TTC)
    per candidate action.
    """
    # Resolve per-agent weight sets.
    _wv = w
    _ws = w_vru if w_vru is not None else w

    col = collision(veh, vru)
    ttc_val = ttc(veh, vru)

    # vehicle cost
    Jv = 0.0
    if col:
        Jv += _wv.w_collision_v
    else:
        if math.isfinite(ttc_val):
            # stronger penalty when TTC small
            Jv += _wv.w_ttc_v / max(ttc_val, 1e-2)

    Jv += _wv.w_speed_v * (veh.v - ref_speed) ** 2
    Jv += _wv.w_acc_v * (a_v.a ** 2)
    jerk = (a_v.a - u_last) / max(veh.dt, 1e-6)
    Jv += _wv.w_jerk_v * (jerk ** 2)

    # vru cost (risk + progress + effort)
    Js = 0.0
    if col:
        Js += _ws.w_collision_s
    else:
        if math.isfinite(ttc_val):
            # penalize small TTC (risk before collision)
            Js += _ws.w_ttc_s / max(ttc_val, 1e-2)

    # progress term: prefer moving (larger speed gives lower cost)
    # speed = math.hypot(vru.vx, vru.vy)
    # Js += -_ws.w_progress_s * speed

    # effort term: yielding / lane change has effort
    effort = 0.0
    if a_s.mode in ("change_left", "change_right"):
        effort += 1.0
    elif a_s.mode == "yield":
        effort += 0.5
    Js += _ws.w_effort_s * effort


    # progress toward goal (more meaningful than speed magnitude)
    gx = getattr(vru, "goal_x", None)
    gy = getattr(vru, "goal_y", None)

    if gx is not None and gy is not None:
        dxg = gx - vru.x
        dyg = gy - vru.y
        ng = math.hypot(dxg, dyg)
        if ng > 1e-6:
            ghatx, ghaty = dxg / ng, dyg / ng
            progress = vru.vx * ghatx + vru.vy * ghaty
        else:
            progress = 0.0
    else:
        # fallback: use speed magnitude
        progress = math.hypot(vru.vx, vru.vy)

    Js += -_ws.w_progress_s * progress

    dbg = {
        "collision": 1.0 if col else 0.0,
        "ttc": ttc_val if math.isfinite(ttc_val) else 1e9,
        "veh_speed_err2": (veh.v - ref_speed) ** 2,
        "veh_acc2": a_v.a ** 2,
        "veh_jerk2": jerk ** 2,
        # "vru_speed": speed,
        "vru_effort": effort,
        "vru_progress": progress,
        "vru_ttc_term": (_ws.w_ttc_s / max(ttc_val, 1e-2)) if (not col and math.isfinite(ttc_val)) else 0.0,
    }
    return Jv, Js, dbg
