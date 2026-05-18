# qual_sim/render_vei_straight_road.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

from traffic_env.traffic_layout import LayoutStyle


# Keep straight-road styling consistent with the intersection renderer.
STRAIGHT_ROAD_STYLE = LayoutStyle(
    lane_solid_color="#455A64",   # blue-gray (dark lane boundaries)
    lane_dash_color="#90A4AE",
    lane_solid_lw=2.2,
    lane_dash_lw=2.0,
    lane_dash_style=(0, (6, 6)),
    show_grid=True,
    tick_labelsize=9,
    tick_labelcolor="#424242",
    show_compass=False,
)


# ------------------------- geometry helpers -------------------------
def _vehicle_polygon_at(
    x: float, y: float, yaw: float,
    length: float, width: float, c2r: float
) -> np.ndarray:
    """
    Return 4x2 polygon vertices (same convention as BicycleModel helper).
    """
    X, Y, Yaw = float(x), float(y), float(yaw)
    L, W, C2R = float(length), float(width), float(c2r)

    offsets = np.array([
        [-C2R, -W/2],
        [-C2R,  W/2],
        [ L-C2R, W/2],
        [ L-C2R,-W/2],
    ]).T  # 2x4

    Rm = np.array([
        [math.cos(Yaw), -math.sin(Yaw)],
        [math.sin(Yaw),  math.cos(Yaw)]
    ])
    verts = (Rm @ offsets) + np.array([[X], [Y]])
    return verts.T  # 4x2


def _get_vehicle_polygon_from_model(veh) -> np.ndarray:
    return _vehicle_polygon_at(
        x=float(veh.x), y=float(veh.y), yaw=float(veh.yaw),
        length=float(veh.LENGTH), width=float(veh.WIDTH), c2r=float(veh.C2R)
    )


def _ttc_ind_to_color(ttc_ind: float) -> str:
    """
    TTC indicator mapping:
      2 safe, 1 attention, 0 alert, -1 collision
    """
    if ttc_ind == 2:
        return "#2ca02c"   # green
    if ttc_ind == 1:
        return "#ffbf00"   # amber
    if ttc_ind == 0:
        return "#d62728"   # red
    if ttc_ind == -1:
        return "#111111"   # black
    return "#7f7f7f"       # unknown/other

# ----------------------- tail coloring (Option 1: TTC bin) -----------------------
from typing import Optional, Sequence

def ttc_bin_to_color(ttc_bin: Optional[int]) -> str:
    if ttc_bin is None:
        return "#9E9E9E"
    try:
        b = int(ttc_bin)
    except Exception:
        b = 0

    if b <= 0:
        return "#E53935"  # red
    if b == 1:
        return "#FDD835"  # yellow
    return "#43A047"      # green


def draw_colored_tail(ax,
                      x_traj: Sequence[float],
                      y_traj: Sequence[float],
                      ttc_bins: Sequence[Optional[int]],
                      tail_len: int,
                      lw: float = 2.8,
                      alpha: float = 0.95,
                      zorder: int = 6):
    """
    Same method as render_vei_intersection.py:
    - draw short segments
    - each segment uses TTC bin at that step
    """
    n = min(len(x_traj), len(y_traj), len(ttc_bins))
    if n < 2:
        return

    tail_len = int(max(2, tail_len))
    i0 = max(0, n - tail_len)

    for i in range(i0, n - 1):
        c = ttc_bin_to_color(ttc_bins[i])
        ax.plot([x_traj[i], x_traj[i + 1]],
                [y_traj[i], y_traj[i + 1]],
                color=c, lw=lw, alpha=alpha, zorder=zorder)

def _build_tail_segments(points_xy: np.ndarray) -> np.ndarray:
    """
    points_xy: (N,2) -> segments: (N-1,2,2)
    """
    if points_xy.shape[0] < 2:
        return np.zeros((0, 2, 2), dtype=float)
    return np.stack([points_xy[:-1], points_xy[1:]], axis=1)


# ------------------------- traffic vehicles (viz only) -------------------------
@dataclass(frozen=True)
class TrafficVehicleCfg:
    x0: float
    y0: float
    v: float = 8.0
    yaw: float = 0.0
    length: float = 4.5
    width: float = 2.0
    color: str = "#666666"
    alpha: float = 0.9
    zorder: int = 2


class TrafficVehicleDrawer:
    """
    Efficient moving vehicles: create patches once, update transform each frame.
    """
    def __init__(self, ax: plt.Axes, vehicles: List[TrafficVehicleCfg]):
        self.ax = ax
        self.vehicles = list(vehicles)
        self.rects: List[patches.Rectangle] = []

        for cfg in self.vehicles:
            rect = patches.Rectangle(
                (-cfg.length / 2.0, -cfg.width / 2.0),
                cfg.length,
                cfg.width,
                facecolor=cfg.color,
                edgecolor="k",
                linewidth=0.8,
                alpha=cfg.alpha,
                zorder=cfg.zorder,
            )
            ax.add_patch(rect)
            self.rects.append(rect)

        self.update(t=0.0)

    def update(self, t: float) -> None:
        for cfg, rect in zip(self.vehicles, self.rects):
            x = float(cfg.x0 + cfg.v * t)
            y = float(cfg.y0)
            yaw = float(cfg.yaw)
            trans = transforms.Affine2D().rotate(yaw).translate(x, y) + self.ax.transData
            rect.set_transform(trans)


# ------------------------- wifi icon (optional) -------------------------
class WifiIcon:
    """
    Patch-based WiFi icon (3 arcs).
    """
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.arcs: List[patches.Arc] = []
        for _ in range(3):
            arc = patches.Arc((0, 0), width=1, height=1, angle=0.0, theta1=20, theta2=160)
            arc.set_visible(False)
            ax.add_patch(arc)
            self.arcs.append(arc)

    def update(self, x: float, y: float, visible: bool, scale: float = 1.0,
               color: str = "blue", alpha: float = 0.55, zorder: int = 10) -> None:
        if not visible:
            for a in self.arcs:
                a.set_visible(False)
            return

        s = float(scale)
        w = 3.0 * s
        h = 2.2 * s
        dy = 0.55 * s
        lw = max(1.0, 2.0 * s)

        arcs = [
            (w,       h,       y + 0.9*s),
            (0.75*w,  0.75*h,  y + 0.9*s - dy),
            (0.50*w,  0.50*h,  y + 0.9*s - 2*dy),
        ]
        for arc_patch, (ww, hh, yc) in zip(self.arcs, arcs):
            arc_patch.center = (float(x), float(yc))
            arc_patch.width = float(ww)
            arc_patch.height = float(hh)
            arc_patch.set_linewidth(lw)
            arc_patch.set_edgecolor(color)
            arc_patch.set_alpha(alpha)
            arc_patch.set_zorder(zorder)
            arc_patch.set_visible(True)


# ------------------------- main renderer -------------------------
class VEIStraightRoadRenderer:
    """
    Renderer-only class (no simulation logic).
    - Draw layout once
    - Update: vehicles, FOV wedge, tails (colored by ttc_ind), panel labels
    """
    def __init__(
        self,
        layout,
        out_dir: str,
        xlim: tuple[float, float] = (-30, 30),
        ylim: tuple[float, float] = (-30, 30),
        tail_len: int = 30,
        dpi: int = 160,
        save_frames: bool = True,
        traffic: Optional[List[TrafficVehicleCfg]] = None,
        show_wifi: bool = True,
    ):
        self.layout = layout
        self.out_dir = out_dir
        self.tail_len = int(tail_len)
        self.dpi = int(dpi)
        self.save_frames = bool(save_frames)
        self.show_wifi = bool(show_wifi)

        os.makedirs(self.out_dir, exist_ok=True)

        self.fig = plt.figure(figsize=(7.2, 7.2))
        self.ax = self.fig.add_subplot(111)

        # draw layout once
        self.ax.clear()
        self.layout.draw(self.ax, style=STRAIGHT_ROAD_STYLE)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect("equal", adjustable="box")

        # dynamic patches placeholders
        self.ego_poly = Polygon(np.zeros((4, 2)), closed=True, facecolor="lightblue", edgecolor="k", lw=1.2, zorder=6)
        self.ax.add_patch(self.ego_poly)
        (self.ego_traj_line,) = self.ax.plot([], [], "-", lw=1.2, alpha=0.90, zorder=3)
        # Trajectory is a stable, single-color line (tail highlights most recent steps).
        self.ego_traj_line.set_color("#90CAF9")

        self.vru_circle = patches.Circle((0, 0), radius=0.3, fill=False, edgecolor="k", lw=1.2, zorder=7)
        self.ax.add_patch(self.vru_circle)
        (self.vru_traj_line,) = self.ax.plot([], [], "-", lw=1.2, alpha=0.85, zorder=3)
        self.vru_traj_line.set_color("#EF9A9A")


        self.fov_wedge = patches.Wedge((0, 0), r=1.0, theta1=0, theta2=0,
                                       facecolor="cyan", alpha=0.20, edgecolor="none", zorder=3)
        self.ax.add_patch(self.fov_wedge)

        # tails (LineCollections)
        self.ego_tail = LineCollection([], linewidths=2.0, zorder=4)
        # self.vru_tail = LineCollection([], linewidths=2.0, zorder=5)
        self.ax.add_collection(self.ego_tail)
        # self.ax.add_collection(self.vru_tail)

        # traffic vehicles
        self.traffic_drawer = None
        if traffic:
            self.traffic_drawer = TrafficVehicleDrawer(self.ax, traffic)

        # wifi icon
        self.wifi = WifiIcon(self.ax) if self.show_wifi else None

        # legend handles (static labels)
        self.ax.plot([], [], color="#90CAF9", lw=1.8, label="ego vehicle")
        self.ax.plot([], [], color="#EF9A9A", lw=1.8, label="e-scooter")
        leg = self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.82),
            bbox_transform=self.ax.transAxes,
            frameon=True,
            fontsize=9,
        )
        leg.get_frame().set_alpha(0.55)
        leg.get_frame().set_linewidth(0.0)

        # panel labels
        kw = dict(transform=self.ax.transAxes, fontsize=10, family="monospace")
        self.txt_lt = self.ax.text(0.02, 0.98, "", va="top", ha="left", **kw)
        self.txt_rt = self.ax.text(0.98, 0.98, "", va="top", ha="right", **kw)
        self.txt_lb = self.ax.text(0.02, 0.01, "", va="bottom", ha="left", **kw)
        self.txt_rb = self.ax.text(0.98, 0.02, "", va="bottom", ha="right", **kw)

        plt.tight_layout()

    def _update_tails(self, veh, vru, ttc_bins) -> None:
        """Match intersection tail logic: color each segment by TTC bin at that step."""
        vx = np.asarray(getattr(veh, "x_traj", []), dtype=float)
        vy = np.asarray(getattr(veh, "y_traj", []), dtype=float)
        sx = np.asarray(getattr(vru, "x_traj", []), dtype=float)
        sy = np.asarray(getattr(vru, "y_traj", []), dtype=float)

        bins = list(ttc_bins) if ttc_bins is not None else []
        if len(bins) == 0:
            bins = [2]

        def build_segments_and_colors(x_arr, y_arr):
            if x_arr.size < 2 or y_arr.size < 2:
                return np.zeros((0, 2, 2)), []

            N = int(min(self.tail_len + 1, x_arr.size, y_arr.size))
            pts = np.stack([x_arr[-N:], y_arr[-N:]], axis=1)
            segs = _build_tail_segments(pts)
            k = segs.shape[0]

            bins_tail = bins[-k:] if len(bins) >= k else ([bins[0]] * k)
            cols = [ttc_bin_to_color(b) for b in bins_tail]
            return segs, cols

        ego_segs, ego_cols = build_segments_and_colors(vx, vy)
        vru_segs, vru_cols = build_segments_and_colors(sx, sy)

        self.ego_tail.set_segments(ego_segs)
        if ego_cols:
            self.ego_tail.set_color(ego_cols)

        # self.vru_tail.set_segments(vru_segs)
        # if vru_cols:
        #     self.vru_tail.set_color(vru_cols)

    def update(
        self,
        frame_idx: int,
        veh,
        vru,
        veh_sensor,
        metric,
        *,
        ttc_bin_hist=None,
        v2x_profile: str,
        rider_type: str,
        a_cmd: float,
        delta_cmd: float,
        vru_mode: str,
        fov_flag: int,
        v2x_flag: int,
        detected: int,
        controller: str,
    ) -> None:
        t = float(veh.t)

        # traffic
        if self.traffic_drawer is not None:
            self.traffic_drawer.update(t=t)

        # ego
        self.ego_poly.set_xy(_get_vehicle_polygon_from_model(veh))

        # vru
        self.vru_circle.center = (float(vru.x), float(vru.y))
        self.vru_circle.set_radius(float(vru.R))

        # fov
        yaw_deg = math.degrees(float(veh.yaw))
        half_deg = math.degrees(float(veh_sensor.th_max))
        self.fov_wedge.set_center((float(veh.x), float(veh.y)))
        self.fov_wedge.set_radius(float(veh_sensor.R_max))
        self.fov_wedge.set_theta1(yaw_deg - half_deg)
        self.fov_wedge.set_theta2(yaw_deg + half_deg)
        self.fov_wedge.set_visible(True)

        # wifi icon (optional)
        if self.wifi is not None:
            self.wifi.update(float(veh.x), float(veh.y) + 3.0, visible=(int(v2x_flag) == 1))

        # ego vehicle trajectory (always visible)
        ex_traj = np.asarray(getattr(veh, "x_traj", []), dtype=float)
        ey_traj = np.asarray(getattr(veh, "y_traj", []), dtype=float)
        if ex_traj.size >= 1 and ey_traj.size >= 1:
            n = int(min(ex_traj.size, ey_traj.size))
            # Show full history for trajectory (tail handles the recent highlight window).
            self.ego_traj_line.set_data(ex_traj[:n], ey_traj[:n])
        else:
            self.ego_traj_line.set_data([], [])
        self.ego_traj_line.set_visible(True)

        # VRU trajectory (always visible)
        x_traj = np.asarray(getattr(vru, "x_traj", []), dtype=float)
        y_traj = np.asarray(getattr(vru, "y_traj", []), dtype=float)
        if x_traj.size >= 1 and y_traj.size >= 1:
            n = int(min(x_traj.size, y_traj.size))
            self.vru_traj_line.set_data(x_traj[:n], y_traj[:n])
        else:
            self.vru_traj_line.set_data([], [])
        self.vru_traj_line.set_visible(True)

        # tails
        bins = ttc_bin_hist
        if bins is None:
            bins = getattr(metric, "ttc_ind_traj", [])
        self._update_tails(veh, vru, bins)


        # panel labels
        vru_speed = math.hypot(float(vru.v_x), float(vru.v_y))
        last_ttc_ind = getattr(metric, "ttc_ind_traj", [None])[-1]
        self.txt_lt.set_text("\n".join([
            f"t = {t:6.2f} s",
            # f"V2X profile = {v2x_profile}",
            f"rider = {rider_type}",
        ]))
        self.txt_rt.set_text("\n".join([
            "EGO VEHICLE",
            f"x = {float(veh.x):7.2f} m",
            f"y = {float(veh.y):7.2f} m",
            f"v = {float(veh.v):6.2f} m/s",
            f"$\\alpha_{{cmd}}$ = {a_cmd:6.2f} m/s$^2$",
            f"$\\delta_{{cmd}}$ = {delta_cmd:6.2f} rad/s",
            f"ctrl = {controller}"
        ]))
        self.txt_lb.set_text("\n".join([
            "E-SCOOTER",
            f"x = {float(vru.x):7.2f} m",
            f"y = {float(vru.y):7.2f} m",
            f"speed = {vru_speed:6.2f} m/s",
            f"mode = {vru_mode}",
            "",
            "OBSERVATION",
            # f"FOV = {int(fov_flag)}",
            # f"V2X = {int(v2x_flag)}",
            f"det = {int(detected)}",
        ]))
        self.txt_rb.set_text("\n".join([
            "EVALUATION",
            f"safety = {float(metric.safety_index):6.3f}",
            f"ttc_ind = {last_ttc_ind}",
            f"collision = {int(getattr(metric, 'has_collision', False))}",
        ]))

        # save
        if self.save_frames:
            self.fig.savefig(os.path.join(self.out_dir, f"Frame{int(frame_idx):04d}.png"), dpi=self.dpi)

        plt.pause(0.001)

    def close(self) -> None:
        plt.close(self.fig)