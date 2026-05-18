# interaction/game/state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


# -------------------------- Params (immutable) --------------------------

@dataclass(frozen=True)
class VehParams:
    WB: float = 2.5
    BACKTOWHEEL: float = 1.0
    WIDTH: float = 2.0
    LENGTH: float = 4.5
    Lr: float = 0.5
    Lf: float = 2.0
    v_max: float = 20.0
    v_min: float = 0.0
    a_max: float = 6.0
    a_min: float = -6.0

    @staticmethod
    def from_bicycle_params(p: Dict[str, float]) -> "VehParams":
        return VehParams(
            WB=float(p.get("WB", 2.5)),
            BACKTOWHEEL=float(p.get("BACKTOWHEEL", 1.0)),
            WIDTH=float(p.get("WIDTH", 2.0)),
            LENGTH=float(p.get("LENGTH", 4.5)),
            Lr=float(p.get("Lr", 0.5)),
            Lf=float(p.get("Lf", 2.0)),
            v_max=float(p.get("v_max", 20.0)),
            v_min=float(p.get("v_min", 0.0)),
            a_max=float(p.get("a_max", 6.0)),
            a_min=float(p.get("a_min", -6.0)),
        )


@dataclass(frozen=True)
class VRUParams:
    m: float
    R: float
    v_max: float
    a_max: float


# -------------------------- States --------------------------

@dataclass
class VehState:
    x: float
    y: float
    psi: float
    v: float
    yaw: float
    beta: float
    u_last: float
    t: float
    dt: float
    params: VehParams

    @staticmethod
    def from_model(veh: Any, params: Optional[VehParams] = None) -> "VehState":
        if params is None:
            params = VehParams(
                WB=float(getattr(veh, "WB", 2.5)),
                BACKTOWHEEL=float(getattr(veh, "BACKTOWHEEL", 1.0)),
                WIDTH=float(getattr(veh, "WIDTH", 2.0)),
                LENGTH=float(getattr(veh, "LENGTH", 4.5)),
                Lr=float(getattr(veh, "Lr", 0.5)),
                Lf=float(getattr(veh, "Lf", 2.0)),
                v_max=float(getattr(veh, "v_max", 20.0)),
                v_min=float(getattr(veh, "v_min", 0.0)),
                a_max=float(getattr(veh, "a_max", 6.0)),
                a_min=float(getattr(veh, "a_min", -6.0)),
            )

        return VehState(
            x=float(getattr(veh, "x", 0.0)),
            y=float(getattr(veh, "y", 0.0)),
            psi=float(getattr(veh, "psi", 0.0)),
            v=float(getattr(veh, "v", 0.0)),
            yaw=float(getattr(veh, "yaw", 0.0)),
            beta=float(getattr(veh, "beta", 0.0)),
            u_last=float(getattr(veh, "u_last", 0.0)),
            t=float(getattr(veh, "t", 0.0)),
            dt=float(getattr(veh, "dt", 0.1)),
            params=params,
        )


@dataclass
class VRUState:
    x: float
    y: float
    vx: float
    vy: float
    t: float
    dt: float
    params: VRUParams

    # (optional) for goal-aligned progress reward later
    goal_x: Optional[float] = None
    goal_y: Optional[float] = None

    @staticmethod
    def from_model(vru: Any, params: Optional[VRUParams] = None) -> "VRUState":
        if params is None:
            params = VRUParams(
                m=float(getattr(vru, "m", 80.0)),
                R=float(getattr(vru, "R", 0.5)),
                v_max=float(getattr(vru, "v_max", 10.0)),
                a_max=float(getattr(vru, "a_max", 2.0)),
            )

        return VRUState(
            x=float(getattr(vru, "x", 0.0)),
            y=float(getattr(vru, "y", 0.0)),
            vx=float(getattr(vru, "v_x", 0.0)),
            vy=float(getattr(vru, "v_y", 0.0)),
            t=float(getattr(vru, "t", 0.0)),
            dt=float(getattr(vru, "dt", 0.1)),
            params=params,
            goal_x=getattr(vru, "goal_x", None),
            goal_y=getattr(vru, "goal_y", None),
        )


# -------------------------- Copy helpers --------------------------

def copy_veh(s: VehState) -> VehState:
    # params is frozen (safe to share)
    return VehState(
        x=s.x, y=s.y, psi=s.psi, v=s.v, yaw=s.yaw, beta=s.beta,
        u_last=s.u_last, t=s.t, dt=s.dt, params=s.params
    )


def copy_vru(s: VRUState) -> VRUState:
    return VRUState(
        x=s.x, y=s.y, vx=s.vx, vy=s.vy, t=s.t, dt=s.dt, params=s.params,
        goal_x=s.goal_x, goal_y=s.goal_y
    )
