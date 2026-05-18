# interaction/game/dynamics.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal
from .state import VehState, VRUState, copy_veh, copy_vru


@dataclass(frozen=True)
class VehicleAction:
    a: float  # longitudinal acceleration command (m/s^2)


@dataclass(frozen=True)
class VRUAction:
    mode: str  # e.g. "go", "yield", "keep", "change_left", "change_right"
    # optional desired lateral velocity for lane change modes
    vy_cmd: Optional[float] = None


def step_vehicle(s: VehState, act: VehicleAction, delta: float) -> VehState:
    """
    Same kinematics as your BicycleModel.update_state (without logs).
    """
    ns = copy_veh(s)
    p = ns.params

    a = float(act.a)
    # clamp accel
    if a >= 0.0:
        a = min(a, p.a_max)
    else:
        a = max(a, p.a_min)

    # beta
    ns.beta = math.atan((p.Lr / (p.Lr + p.Lf)) * math.tan(float(delta)))

    ns.x = ns.x + ns.v * math.cos(ns.psi + ns.beta) * ns.dt
    ns.y = ns.y + ns.v * math.sin(ns.psi + ns.beta) * ns.dt
    ns.psi = ns.psi + (ns.v / p.Lr) * math.sin(ns.beta) * ns.dt
    ns.v = ns.v + a * ns.dt

    # speed constraints
    ns.v = max(p.v_min, min(p.v_max, ns.v))

    ns.t = ns.t + ns.dt
    ns.yaw = ns.beta + ns.psi
    ns.u_last = a
    return ns


def step_vru(s: VRUState, act: VRUAction) -> VRUState:
    """
    Simple VRU dynamics for game layer (explicit, deterministic):
    - "go": keep current velocity
    - "yield": reduce speed (brake) by scaling vx/vy
    - "keep": keep
    - "change_left/right": impose lateral vy_cmd (if provided), otherwise +/- default
    """
    ns = copy_vru(s)

    if act.mode == "go":
        pass

    elif act.mode == "keep":
        pass

    elif act.mode == "yield":
        a_yield = ns.params.a_yield if hasattr(ns.params, "a_yield") else 2.0  # m/s^2
        v = math.hypot(ns.vx, ns.vy)
        v_new = max(0.0, v - a_yield * ns.dt)
        if v > 1e-9:
            scale = v_new / v
            ns.vx *= scale
            ns.vy *= scale

    elif act.mode == "change_left":  # TODO： interpret lane-change as a target lateral velocity controller
        vy = act.vy_cmd if act.vy_cmd is not None else +1.2
        ns.vy = float(vy)

    elif act.mode == "change_right":
        vy = act.vy_cmd if act.vy_cmd is not None else -1.2
        ns.vy = float(vy)

    else:
        raise ValueError(f"Unknown VRUAction.mode='{act.mode}'")

    # speed cap
    v = math.hypot(ns.vx, ns.vy)
    vmax = ns.params.v_max
    if v > vmax and v > 1e-9:
        ns.vx *= vmax / v
        ns.vy *= vmax / v

    # integrate (Euler)
    ns.x += ns.vx * ns.dt
    ns.y += ns.vy * ns.dt
    ns.t += ns.dt
    return ns
