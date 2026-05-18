"""collision_check.py

Reusable, precise collision check between:
  - VRU modeled as a circle (center at vru.(x,y), radius vru.R)
  - Vehicle modeled as an oriented bounding box (OBB) in the vehicle local frame:
        local-x in [-C2R, +C2F]
        local-y in [-WIDTH/2, +WIDTH/2]

This module is intended to be imported by both simulation scripts (qual_sim)
and batch quantification scripts (quant_sim) so collision logic stays consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _as_scalar(v) -> float:
    """Robust scalar conversion (float, np scalar, 0-d/1-d arrays)."""
    a = np.asarray(v)
    return float(a.ravel()[0])


def _vehicle_extents(veh) -> tuple[float, float, float]:
    """Return (c2f, c2r, half_w) for vehicle OBB extents in vehicle-local frame.

    - local x: forward along heading
    - local y: left of heading
    - OBB: x in [-c2r, +c2f], y in [-half_w, +half_w]

    Supports both:
      - attributes on veh: C2F, C2R, LENGTH, WIDTH
      - or on veh.params: C2F, C2R, LENGTH, WIDTH
    """
    params = getattr(veh, "params", None)

    length = float(getattr(veh, "LENGTH", getattr(params, "LENGTH", 0.0)))
    width = float(getattr(veh, "WIDTH", getattr(params, "WIDTH", 0.0)))

    c2f = getattr(veh, "C2F", getattr(params, "C2F", None))
    c2r = getattr(veh, "C2R", getattr(params, "C2R", None))

    if c2f is None or c2r is None:
        # fallback: symmetric rectangle about the rear axle / CoM
        c2f = 0.5 * length
        c2r = 0.5 * length
    else:
        c2f = float(c2f)
        c2r = float(c2r)

    half_w = 0.5 * width
    return c2f, c2r, half_w


def _vru_radius(vru) -> float:
    params = getattr(vru, "params", None)
    r = getattr(vru, "R", getattr(params, "R", None))
    return float(r if r is not None else 0.0)


@dataclass(frozen=True)
class CollisionResult:
    hit: bool
    contact_point: Optional[Tuple[float, float]] = None


def circle_vs_vehicle_obb(veh, vru) -> CollisionResult:
    """Exact circle (VRU) vs oriented-rectangle (vehicle) collision test.

    Returns:
      CollisionResult(hit, contact_point)

    contact_point is the nearest point on the vehicle OBB to the circle center,
    in world coordinates (only provided when hit=True).
    """
    cx = _as_scalar(vru.x)
    cy = _as_scalar(vru.y)
    r = _vru_radius(vru)

    vx = _as_scalar(veh.x)
    vy = _as_scalar(veh.y)
    yaw = _as_scalar(getattr(veh, "yaw", 0.0))

    c2f, c2r, half_w = _vehicle_extents(veh)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    # --- broad-phase AABB of rotated car (padded by r)
    front = (c2f * cos_y, c2f * sin_y)
    rear = (-c2r * cos_y, -c2r * sin_y)
    width_vec = (-half_w * sin_y, half_w * cos_y)

    fl = (vx + front[0] + width_vec[0], vy + front[1] + width_vec[1])
    fr = (vx + front[0] - width_vec[0], vy + front[1] - width_vec[1])
    rl = (vx + rear[0] + width_vec[0], vy + rear[1] + width_vec[1])
    rr = (vx + rear[0] - width_vec[0], vy + rear[1] - width_vec[1])

    xs = [fl[0], fr[0], rl[0], rr[0]]
    ys = [fl[1], fr[1], rl[1], rr[1]]

    if not (min(xs) - r <= cx <= max(xs) + r and min(ys) - r <= cy <= max(ys) + r):
        return CollisionResult(False, None)

    # --- narrow-phase: circle center -> vehicle local frame
    dx = cx - vx
    dy = cy - vy
    lx = cos_y * dx + sin_y * dy
    ly = -sin_y * dx + cos_y * dy

    # clamp to rectangle to find nearest point
    clx = min(max(lx, -c2r), c2f)
    cly = min(max(ly, -half_w), half_w)

    dist2 = (lx - clx) ** 2 + (ly - cly) ** 2
    hit = dist2 <= (r + 1e-9) ** 2
    if not hit:
        return CollisionResult(False, None)

    # nearest point back to world
    px = vx + cos_y * clx - sin_y * cly
    py = vy + sin_y * clx + cos_y * cly
    return CollisionResult(True, (float(px), float(py)))


def check_collision(veh, vru) -> bool:
    """Return True if VRU circle intersects vehicle OBB."""
    return bool(circle_vs_vehicle_obb(veh, vru).hit)


class VRUCollisionCheck:
    """Convenience tracker (optional) matching your earlier API style."""

    def __init__(self, vru, dt: float, t0: float):
        self.vru = vru
        self.dt = float(dt)
        self.t = float(t0)
        self.collision_state = "False"
        self.collision_ind_state = 0
        self.collision_state_traj: list[int] = []
        self.contact_point: Optional[Tuple[float, float]] = None

    def update(self, veh):
        hit, pt = circle_vs_vehicle_obb(veh, self.vru).hit, circle_vs_vehicle_obb(veh, self.vru).contact_point
        self.collision_state = "True" if hit else "False"
        self.collision_ind_state = 1 if hit else 0
        self.collision_state_traj.append(self.collision_ind_state)
        self.contact_point = pt
        return hit
