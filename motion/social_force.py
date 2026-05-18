# motion/social_force.py
from __future__ import annotations
import math
from typing import Tuple

def sfm_force_to_target(
    vru,
    target_xy: Tuple[float, float],
    v_des: float,
    tau: float = 0.6,
) -> Tuple[float, float]:
    """
    Minimal Social-Force / Desired-velocity force (deterministic, easy to debug):

    desired_velocity = v_des * unit(target - pos)
    a_des = (desired_velocity - current_velocity) / tau
    F = m * a_des

    NOTE:
    - No noise, no delay.
    - Saturation is handled inside PointMassNewton.update() already (m*a_max).
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    dx = tx - float(vru.x)
    dy = ty - float(vru.y)
    dist = math.hypot(dx, dy)

    if dist < 1e-6:
        # already at target => desire zero velocity
        vdx, vdy = 0.0, 0.0
    else:
        ux, uy = dx / dist, dy / dist
        vdx, vdy = v_des * ux, v_des * uy

    ax = (vdx - float(vru.v_x)) / max(tau, 1e-3)
    ay = (vdy - float(vru.v_y)) / max(tau, 1e-3)

    Fx = float(vru.m) * ax
    Fy = float(vru.m) * ay
    return Fx, Fy
