# qual_sim/render_vei_intersection.py
from __future__ import annotations

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from typing import Optional, Sequence
from traffic_env.traffic_layout import LayoutStyle

# ----------------------- geometry helpers (render-only) -----------------------
INTERSECTION_STYLE = LayoutStyle(
    lane_solid_color="#455A64",   # example: blue-gray
    lane_dash_color="#90A4AE",
    lane_solid_lw=2.2,
    lane_dash_lw=2.0,
    lane_dash_style=(0, (6, 6)),  # nicer dash pattern than "--"
    show_grid=True,
    tick_labelsize=9,
    tick_labelcolor="#424242",
    show_compass=False,
)

def _get_vehicle_polygon(veh):
    """Return polygon vertices for plotting."""
    C2R = float(veh.C2R)
    L = float(veh.LENGTH)
    W = float(veh.WIDTH)
    X, Y, Yaw = float(veh.x), float(veh.y), float(veh.yaw)

    offsets = np.array([
        [-C2R, -W / 2],
        [-C2R,  W / 2],
        [ L - C2R,  W / 2],
        [ L - C2R, -W / 2],
    ]).T  # 2x4

    Rm = np.array([[math.cos(Yaw), -math.sin(Yaw)],
                   [math.sin(Yaw),  math.cos(Yaw)]])
    verts = (Rm @ offsets) + np.array([[X], [Y]])
    return verts.T  # 4x2


def _vehicle_polygon_at(x: float, y: float, yaw: float,
                        length: float, width: float, c2r: float):
    """Build a rectangle polygon for a static vehicle."""
    X, Y, Yaw = float(x), float(y), float(yaw)
    L, W, C2R = float(length), float(width), float(c2r)

    offsets = np.array([
        [-C2R, -W / 2],
        [-C2R,  W / 2],
        [ L - C2R,  W / 2],
        [ L - C2R, -W / 2],
    ]).T  # 2x4

    Rm = np.array([[math.cos(Yaw), -math.sin(Yaw)],
                   [math.sin(Yaw),  math.cos(Yaw)]])
    verts = (Rm @ offsets) + np.array([[X], [Y]])
    return verts.T  # 4x2


def draw_static_vehicle(ax, x: float, y: float, yaw: float,
                        length: float = 4.5, width: float = 2.0,
                        c2r: float = 1.5,
                        facecolor: str = "#B0B0B0",
                        edgecolor: str = "k",
                        alpha: float = 0.9,
                        zorder: int = 2):
    poly = _vehicle_polygon_at(x, y, yaw, length, width, c2r)
    patch = Polygon(poly, closed=True,
                    facecolor=facecolor, edgecolor=edgecolor,
                    lw=1.0, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


def draw_wifi_icon(ax, x, y, scale=1.0, color="blue", alpha=0.6, zorder=10):
    """Simple WiFi icon using arcs (visual only)."""
    x = float(x); y = float(y); s = float(scale)
    w = 3.0 * s
    h = 2.2 * s
    dy = 0.55 * s
    lw = max(1.0, 2.0 * s)

    arcs = [
        (w,       h,       y + 0.9 * s),
        (0.75*w,  0.75*h,  y + 0.9 * s - dy),
        (0.50*w,  0.50*h,  y + 0.9 * s - 2 * dy),
    ]
    for ww, hh, yc in arcs:
        arc = patches.Arc(
            (x, yc),
            width=ww, height=hh,
            angle=0.0,
            theta1=20, theta2=160,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(arc)


# ----------------------- tail coloring (Option 1: TTC bin) -----------------------
def ttc_bin_to_color(ttc_bin: Optional[int]) -> str:
    """
    Map TTC indicator bin to a color.
    You can adjust these mapping choices in ONE place.
    """
    if ttc_bin is None:
        return "#9E9E9E"  # unknown
    try:
        b = int(ttc_bin)
    except Exception:
        b = 0

    # Typical convention: larger bin => safer
    if b <= 0:
        return "#E53935"  # red (danger)
    if b == 1:
        return "#FDD835"  # yellow (caution)
    return "#43A047"      # green (safe)


def draw_colored_tail(ax,
                      x_traj: Sequence[float],
                      y_traj: Sequence[float],
                      ttc_bins: Sequence[Optional[int]],
                      tail_len: int,
                      lw: float = 2.8,
                      alpha: float = 0.95,
                      zorder: int = 6):
    """
    Draw a tail using small line segments. Each segment uses the TTC-bin at that time step.
    Assumes ttc_bins is per-step (same cadence as appended each loop).
    We align lengths robustly via the minimum available length.
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


# ----------------------- renderer class -----------------------
class VEIIntersectionRenderer:
    """
    All plotting code lives here.
    sim_vei_intersection.py should only call renderer.draw_frame(...) each step.
    """

    def __init__(self,
                 layout,
                 veh_sensor,
                 out_dir: str,
                 save_frames: bool = True,
                 xlim=(-30, 30),
                 ylim=(-30, 30),
                 figsize=(7.2, 7.2),
                 dpi: int = 160,
                 tail_len: int = 30,
                 show_full_traj: bool = True):

        self.layout = layout
        self.veh_sensor = veh_sensor
        self.out_dir = out_dir
        self.save_frames = bool(save_frames)
        self.xlim = xlim
        self.ylim = ylim
        self.dpi = int(dpi)
        self.tail_len = int(tail_len)
        self.show_full_traj = bool(show_full_traj)

        os.makedirs(self.out_dir, exist_ok=True)

        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        # Text style knob (change here anytime)
        self.kw = dict(transform=self.ax.transAxes, fontsize=10, family="monospace")

    def close(self):
        plt.ioff()
        plt.close(self.fig)

    def draw_frame(self,
                   step: int,
                   veh,
                   vru,
                   vru_mode: str,
                   fov_flag: int,
                   v2x_detected: int,
                   detected: int,
                   metric,
                   a_cmd: float,
                   delta_cmd: float,
                   v2x_profile: str,
                   rider_type: str,
                   ttc_bin_hist: Sequence[Optional[int]],
                   controller: str):

        ax = self.ax
        ax.clear()
        self.layout.draw(ax, style=INTERSECTION_STYLE)

        # --- parked vehicles (visual only) ---
        draw_static_vehicle(ax, x=2.0, y=-20.0, yaw=math.pi / 2, facecolor="#808080", alpha=0.8)
        draw_static_vehicle(ax, x=6.0, y=-20.0, yaw=math.pi / 2, facecolor="#808080", alpha=0.8)

        # --- trajectories (full + tail) ---
        if self.show_full_traj:
            ax.plot(veh.x_traj, veh.y_traj, color="#90CAF9", lw=1.2, alpha=0.55, label="ego vehicle")
            ax.plot(vru.x_traj, vru.y_traj, color="#EF9A9A", lw=1.2, alpha=0.55, label="e-scooter")

        # Tail colored by TTC-bin (Option 1)
        draw_colored_tail(ax, veh.x_traj, veh.y_traj, ttc_bin_hist, tail_len=self.tail_len, lw=3.0, alpha=0.95)
        # draw_colored_tail(ax, vru.x_traj, vru.y_traj, ttc_bin_hist, tail_len=self.tail_len, lw=3.0, alpha=0.95)

        # --- shapes ---
        veh_poly = Polygon(_get_vehicle_polygon(veh), closed=True,
                           facecolor="lightblue", edgecolor="k", lw=1.2, zorder=7)
        ax.add_patch(veh_poly)

        vru_circle = patches.Circle((vru.x, vru.y), radius=vru.R,
                                    fill=False, edgecolor="k", lw=1.2, zorder=7)
        ax.add_patch(vru_circle)

        # --- FOV ---
        yaw_deg = math.degrees(float(veh.yaw))
        half_deg = math.degrees(float(self.veh_sensor.th_max))
        fov_patch = patches.Wedge(
            center=(veh.x, veh.y),
            r=float(self.veh_sensor.R_max),
            theta1=yaw_deg - half_deg,
            theta2=yaw_deg + half_deg,
            facecolor="cyan",
            alpha=0.20,
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(fov_patch)

        # V2X icon (you’ll remove later in Phase 3; keeping for now)
        if int(v2x_detected) == 1:
            draw_wifi_icon(ax, veh.x, veh.y + 3.0, scale=1.0, color="blue", alpha=0.55)

        # ---- panel text ----
        kw = self.kw
        ax.text(
            0.02, 0.98,
            "\n".join([
                f"t = {veh.t:6.2f} s",
                # f"V2X profile = {v2x_profile}",
                f"rider = {rider_type}",
            ]),
            va="top", ha="left", **kw
        )

        ax.text(
            0.98, 0.98,
            "\n".join([
                "EGO VEHICLE",
                f"x = {veh.x:7.2f} m",
                f"y = {veh.y:7.2f} m",
                f"v = {veh.v:6.2f} m/s",
                # f"yaw = {veh.yaw:6.2f} rad",
                f"$\\alpha_{{cmd}}$ = {a_cmd:6.2f} m/s$^2$",
                f"$\\delta_{{cmd}}$ = {delta_cmd:6.2f} rad/s",
                f"ctrl = {controller}",
            ]),
            va="top", ha="right", **kw
        )

        vru_speed = math.hypot(vru.v_x, vru.v_y)
        ax.text(
            0.02, 0.01,
            "\n".join([
                "E-SCOOTER",
                f"x = {vru.x:7.2f} m",
                f"y = {vru.y:7.2f} m",
                f"speed = {vru_speed:6.2f} m/s",
                f"mode = {vru_mode}",
                "",
                "OBSERVATION",
                # f"FOV = {int(fov_flag)}",
                # f"V2X = {int(v2x_detected)}",
                f"det = {int(detected)}",
            ]),
            va="bottom", ha="left", **kw
        )

        # show TTC bin on panel (so you know what tail color means)
        cur_bin = ttc_bin_hist[-1] if len(ttc_bin_hist) > 0 else None
        ax.text(
            0.98, 0.02,
            "\n".join([
                "EVALUATION",
                f"safety score = {metric.safety_index:6.3f}",
                f"ttc_ind bin  = {str(cur_bin)}",
                f"collision    = {int(getattr(metric, 'collision_state', 0)) if hasattr(metric, 'collision_state') else ''}".strip(),
            ]),
            va="bottom", ha="right", **kw
        )

        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal", adjustable="box")

        # legend placement
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.82),
            bbox_transform=ax.transAxes,
            frameon=True,
            fontsize=9,
        )
        leg.get_frame().set_alpha(0.55)
        leg.get_frame().set_linewidth(0.0)

        plt.pause(0.001)

        if self.save_frames:
            self.fig.savefig(os.path.join(self.out_dir, f"Frame{step:04d}.png"), dpi=self.dpi)
