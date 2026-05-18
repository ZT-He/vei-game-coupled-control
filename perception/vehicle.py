import numpy as np
import math
from typing import List, Tuple, Optional

class PredictorLinear:
    def __init__(self, dt, t0):
        self.t = t0
        self.dt = dt

        # record
        self.t_traj = []
        self.pred_traj = []

    def predict(self, traj_past, t_pred, radius):
        sx = float(traj_past[0])
        sy = float(traj_past[1])
        vx = float(traj_past[2])
        vy = float(traj_past[3])

        l_pred = int(t_pred / self.dt)
        traj_pred = np.empty((l_pred+1, 5))
        for i in range(l_pred+1):
            traj_pred[i, :] = np.array([sx + i * vx * self.dt, sy + i * vy * self.dt, vx, vy, radius])

        self.t_traj.append(self.t)
        self.pred_traj.append(traj_pred)
        self.t = self.t + self.dt

        return traj_pred


Pt = Tuple[float, float]
Poly = List[Pt]

class VehFOV:
    """
    Simple sector-FOV perception model with optional occlusion.

    raw detection = 1 if VRU circle intersects sector (no occlusion)
    final detection = raw AND (not occluded)

    Occlusion rule:
      If line segment from ego (veh.x,veh.y) to VRU (vru.x,vru.y)
      intersects ANY occluder polygon -> occluded -> final detection = 0
    """

    def __init__(self, fov_range, fov_angle, vehicle, vru):
        self.R_max  = float(fov_range)            # [m]
        self.th_max = float(fov_angle) / 2.0      # half-angle [rad]
        self.veh    = vehicle
        self.vru    = vru
        self.dt     = vehicle.dt
        self.t      = vehicle.t

        # history log
        self.hist_t   = []
        self.hist_d   = []   # final detection (after occlusion)
        self.hist_raw = []   # sector-only detection
        self.hist_occ = []   # 1 if occluded else 0

    # --------------------------- sector check ---------------------------
    def _point_in_sector(self, dx, dy, R_vru):
        dist = math.hypot(dx, dy)
        if dist > self.R_max + R_vru:
            return 0

        ang = math.atan2(dy, dx) - float(self.veh.yaw)
        ang = (ang + math.pi) % (2 * math.pi) - math.pi

        ang_margin = self.th_max + math.asin(min(1.0, R_vru / max(dist, 1e-6)))
        return 1 if (abs(ang) <= ang_margin and dist <= self.R_max + R_vru) else 0

    # --------------------------- geometry helpers ---------------------------
    @staticmethod
    def _orient(a: Pt, b: Pt, c: Pt) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def _on_segment(a: Pt, b: Pt, p: Pt, eps: float = 1e-9) -> bool:
        # collinear + inside bounding box
        if abs(VehFOV._orient(a, b, p)) > eps:
            return False
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        )

    @staticmethod
    def _seg_intersect(p1: Pt, p2: Pt, q1: Pt, q2: Pt, eps: float = 1e-9) -> bool:
        o1 = VehFOV._orient(p1, p2, q1)
        o2 = VehFOV._orient(p1, p2, q2)
        o3 = VehFOV._orient(q1, q2, p1)
        o4 = VehFOV._orient(q1, q2, p2)

        # Proper intersection
        if (o1 * o2 < -eps) and (o3 * o4 < -eps):
            return True

        # Touching / collinear cases
        if abs(o1) <= eps and VehFOV._on_segment(p1, p2, q1, eps): return True
        if abs(o2) <= eps and VehFOV._on_segment(p1, p2, q2, eps): return True
        if abs(o3) <= eps and VehFOV._on_segment(q1, q2, p1, eps): return True
        if abs(o4) <= eps and VehFOV._on_segment(q1, q2, p2, eps): return True

        return False

    @staticmethod
    def _seg_intersects_poly(p1: Pt, p2: Pt, poly: Poly) -> bool:
        n = len(poly)
        if n < 2:
            return False
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if VehFOV._seg_intersect(p1, p2, a, b):
                return True
        return False

    @staticmethod
    def _is_occluded(ego_xy: Pt, vru_xy: Pt, occluder_polys: List[Poly]) -> bool:
        for poly in occluder_polys:
            if VehFOV._seg_intersects_poly(ego_xy, vru_xy, poly):
                return True
        return False

    # --------------------------- update ---------------------------
    def update(self, occluder_polys: Optional[List[Poly]] = None, return_details: bool = False):
        dx = float(self.vru.x) - float(self.veh.x)
        dy = float(self.vru.y) - float(self.veh.y)

        raw = int(self._point_in_sector(dx, dy, float(self.vru.R)))

        occ = 0
        det = raw

        # Only run occlusion test if raw says "in FOV"
        if raw == 1 and occluder_polys:
            ego_xy = (float(self.veh.x), float(self.veh.y))
            vru_xy = (float(self.vru.x), float(self.vru.y))
            if self._is_occluded(ego_xy, vru_xy, occluder_polys):
                occ = 1
                det = 0

        # log
        self.hist_t.append(float(self.t))
        self.hist_raw.append(int(raw))
        self.hist_occ.append(int(occ))
        self.hist_d.append(int(det))
        self.t += float(self.dt)

        if return_details:
            return {"detected": int(det), "raw": int(raw), "occluded": int(occ)}
        return int(det)