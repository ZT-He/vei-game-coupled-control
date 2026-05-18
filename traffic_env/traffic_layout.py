import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Optional

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class LayoutStyle:
    # lane appearance
    lane_solid_color: str = "#2E7D32"   # default ~ your green
    lane_dash_color: str = "#2E7D32"
    lane_solid_lw: float = 2.0
    lane_dash_lw: float = 2.0
    lane_dash_style: str = "--"         # matplotlib linestyle

    # grid / background
    show_grid: bool = True
    grid_color: str = "#BDBDBD"
    grid_alpha: float = 0.35
    grid_lw: float = 0.8

    # axis tick labels (what you called “labels” in layout)
    tick_labelsize: int = 9
    tick_labelcolor: str = "#424242"

    # optional compass labels (N/E/S/W), can turn on later
    show_compass: bool = False
    compass_fontsize: int = 12
    compass_color: str = "#212121"
    compass_family: str = "sans-serif"


class TrafficLayout:
    """
    Encapsulates static traffic layouts for the VEI simulation.

    Available layouts:
      - 'vei_intersection'
      - 'vei_straight_road'
      - 'vru_euroncap_intersec'
      - 'vru_euroncap_straightroad'
    """

    def __init__(
        self,
        name: str,
        X_min: float, X_max: float,
        Y_min: float, Y_max: float,
        W_lane: float,
        num_lanes: int = 2,
        style: Optional[LayoutStyle] = None
    ):
        self.X_min = X_min
        self.X_max = X_max
        self.Y_min = Y_min
        self.Y_max = Y_max
        self.W_lane = W_lane
        self.num_lanes = num_lanes
        self.style = style or LayoutStyle()

        layouts = {
            'vei_intersection':      self._draw_vei_intersection,
            'vei_straight_road':     self._draw_vei_straight_road,
            'vru_euroncap_intersec': self._draw_vru_euroncap_intersection,
            'vru_euroncap_strrd': self._draw_vru_euroncap_straight_road,
        }
        try:
            self._draw_fn = layouts[name]
        except KeyError:
            raise ValueError(
                f"Unknown layout '{name}'. Valid options: {list(layouts)}"
            )

    def draw(self, ax: plt.Axes, style: Optional[LayoutStyle] = None):
        st = style or self.style

        if st.show_grid:
            ax.grid(True, color=st.grid_color, alpha=st.grid_alpha, linewidth=st.grid_lw)
        else:
            ax.grid(False)

        # axis tick label styling
        ax.tick_params(axis="both", labelsize=st.tick_labelsize, colors=st.tick_labelcolor)

        self._draw_fn(ax, st)
        ax.set_aspect("equal")

        if st.show_compass:
            self._draw_compass(ax, st)

    def _draw_vei_intersection(self, ax: plt.Axes, st: LayoutStyle):
        """
        VEI intersection with 4-lane road (two-way traffic):
          - total 4 lanes per road segment (2 lanes per direction)
          - north/south/east/west approaches
          - rounded corners similar to EuroNCAP style (fillet arcs)
        Geometry (in terms of lane width W):
          - lane boundaries at y = ±2W, ±W, 0 for horizontal road
          - lane boundaries at x = ±2W, ±W, 0 for vertical road
          - "rounded" corners connect the outer boundaries (±2W) using radius R
        """
        X0, X1 = float(self.X_min), float(self.X_max)
        Y0, Y1 = float(self.Y_min), float(self.Y_max)
        W = float(self.W_lane)

        # 4-lane road total half-width = 2W (two lanes each direction)
        half_width = 2.0 * W

        # fillet radius (tunable): must be > half_width to look good
        R = 8.0

        # small visual offsets (same idea as EuroNCAP drawing)
        vo1 = 0.05
        vo2 = 0.06
        vo3 = 0.01

        # -----------------------
        # 1) Outer solid boundaries (4 boundaries: y=±2W, x=±2W)
        #    Draw them as straight segments that stop before the fillet region.
        # -----------------------
        # Horizontal outer boundaries: y = +2W and y = -2W
        y_up = +half_width + vo1
        y_dn = -half_width + vo1

        # Vertical outer boundaries: x = +2W and x = -2W
        x_rt = +half_width - vo2 + vo3
        x_lt = -half_width - vo2 - vo3

        # stop points must match the *actual* outer boundary locations (with offsets)
        x_stop_left = x_lt - R
        x_stop_right = x_rt + R
        y_stop_bot = y_dn - R
        y_stop_top = y_up + R

        ax.plot([X0, x_stop_left], [y_up, y_up], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')
        ax.plot([x_stop_right, X1], [y_up, y_up], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')

        ax.plot([X0, x_stop_left], [y_dn, y_dn], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')
        ax.plot([x_stop_right, X1], [y_dn, y_dn], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')

        ax.plot([x_lt, x_lt], [Y0, y_stop_bot], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')
        ax.plot([x_lt, x_lt], [y_stop_top, Y1], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')

        ax.plot([x_rt, x_rt], [Y0, y_stop_bot], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')
        ax.plot([x_rt, x_rt], [y_stop_top, Y1], color=st.lane_solid_color, lw=st.lane_solid_lw, solid_capstyle='round', solid_joinstyle='round')

        # -----------------------
        # 2) Rounded corner arcs that connect outer boundaries (like EuroNCAP)
        #    Outer boundary "corner centers" are at (±(2W+R), ±(2W+R))
        # -----------------------
        def draw_arc(cx, cy, th1, th2):
            thetas = np.linspace(th1, th2, 120)
            x = cx + R * np.cos(thetas)
            y = cy + R * np.sin(thetas)
            ax.plot(x, y, color=st.lane_solid_color, lw=st.lane_solid_lw,
                    solid_capstyle='round',
                    solid_joinstyle='round')

        # four fillet arcs (connect the outer boundary box) -- consistent with offsets
        arcs = [
            (x_stop_right, y_stop_top, np.pi, 3 * np.pi / 2),  # NE
            (x_stop_left, y_stop_top, 3 * np.pi / 2, 2 * np.pi),  # NW
            (x_stop_left, y_stop_bot, 0.0, np.pi / 2),  # SW
            (x_stop_right, y_stop_bot, np.pi / 2, np.pi),  # SE
        ]
        for cx, cy, t1, t2 in arcs:
            draw_arc(cx, cy, t1, t2)

        # -----------------------
        # 3) Lane separators (dashed): y=±W and y=0 ; x=±W and x=0
        #    We only draw them on the approach arms (not through the fillet arcs)
        # -----------------------
        # Horizontal dashed lines: y = +W, 0, -W
        for y_mid in (+W, 0.0, -W):
            y_mid = y_mid + vo1
            ax.plot([X0, x_stop_left], [y_mid, y_mid], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)
            ax.plot([x_stop_right, X1], [y_mid, y_mid], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)

        # Vertical dashed lines: x = +W, 0, -W
        for x_mid in (+W, 0.0, -W):
            x_mid = x_mid - vo2 + (vo3 if x_mid > 0 else -vo3 if x_mid < 0 else 0.0)
            ax.plot([x_mid, x_mid], [Y0, y_stop_bot], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)
            ax.plot([x_mid, x_mid], [y_stop_top, Y1], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)

    def _draw_vei_straight_road(self, ax: plt.Axes, st: LayoutStyle):
        X0, X1 = self.X_min, self.X_max
        Y0, Y1 = self.Y_min, self.Y_max
        W      = self.W_lane
        n      = self.num_lanes

        left  = - (n/2) * W
        right =  (n/2) * W

        # solid outer boundaries
        ax.vlines(left,  Y0, Y1, color=st.lane_solid_color, lw=st.lane_solid_lw)
        ax.vlines(right, Y0, Y1, color=st.lane_solid_color, lw=st.lane_solid_lw)
        # dashed separators
        for i in range(1, n):
            x = (-n/2 + i) * W
            ax.vlines(x, Y0, Y1, color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)

    def _draw_vru_euroncap_intersection(self, ax: plt.Axes,st: LayoutStyle):
        X0, X1 = self.X_min, self.X_max
        Y0, Y1 = self.Y_min, self.Y_max
        W      = self.W_lane
        R      = 8.0

        visual_offset_1 = 0.05
        visual_offset_2 = 0.06
        visual_offset_3 = 0.01

        # draw connecting horizontal & vertical segments
        # top approach
        ax.plot([X0, -(W+R)], [ W + visual_offset_1,  W + visual_offset_1], color=st.lane_solid_color, lw=st.lane_solid_lw)
        ax.plot([ W+R, X1],  [ W + visual_offset_1,  W + visual_offset_1], color=st.lane_solid_color, lw=st.lane_solid_lw)
        # bottom approach
        ax.plot([X0, -(W+R)], [-W + visual_offset_1, -W + visual_offset_1], color=st.lane_solid_color, lw=st.lane_solid_lw)
        ax.plot([ W+R, X1],  [-W + visual_offset_1, -W + visual_offset_1], color=st.lane_solid_color, lw=st.lane_solid_lw)
        # left approach
        ax.plot([-W - visual_offset_2 - visual_offset_3, -W - visual_offset_2 - visual_offset_3], [Y0, -(W+R)], color=st.lane_solid_color, lw=st.lane_solid_lw)
        ax.plot([-W - visual_offset_2 - visual_offset_3, -W - visual_offset_2 - visual_offset_3], [ W+R, Y1], color=st.lane_solid_color, lw=st.lane_solid_lw)
        # right approach
        ax.plot([ W - visual_offset_2 + visual_offset_3,  W - visual_offset_2 + visual_offset_3], [Y0, -(W+R)], color=st.lane_solid_color, lw=st.lane_solid_lw)
        ax.plot([ W - visual_offset_2 + visual_offset_3,  W - visual_offset_2 + visual_offset_3], [ W+R, Y1], color=st.lane_solid_color, lw=st.lane_solid_lw)

        # helper to draw a smooth corner arc
        def draw_arc(cx, cy, th1, th2):
            thetas = np.linspace(th1, th2, 80)
            x = cx + R * np.cos(thetas)
            y = cy + R * np.sin(thetas)
            ax.plot(x, y, color=st.lane_solid_color, lw=st.lane_solid_lw)

        # four fillet arcs
        arcs = [
            ( W+R,  W+R,  np.pi,     3*np.pi/2),  # top-right
            (-W-R,  W+R,  3*np.pi/2, 2*np.pi),    # top-left
            (-W-R, -W-R,  0,         np.pi/2),    # bottom-left
            ( W+R, -W-R,  np.pi/2,   np.pi),      # bottom-right
        ]
        for cx, cy, t1, t2 in arcs:
            draw_arc(cx, cy, t1, t2)

        # center dashed lines on approach arms
        ax.plot([X0,    -(W+R)], [0, 0], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)
        ax.plot([ W+R,   X1   ], [0, 0], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)
        ax.plot([0, 0], [Y0,   -(W+R)], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)
        ax.plot([0, 0], [ W+R,  Y1    ], color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)


    def _draw_vru_euroncap_straight_road(self, ax: plt.Axes, st: LayoutStyle):
        X0, X1 = self.X_min, self.X_max
        Y0, Y1 = self.Y_min, self.Y_max
        W      = self.W_lane
        n      = self.num_lanes

        up   =    (n/2) * W
        down = - (n/2) * W

        # solid outer boundaries
        ax.hlines(up,  X0, X1, color=st.lane_solid_color, lw=st.lane_solid_lw)
        ax.hlines(down, X0, X1, color=st.lane_solid_color, lw=st.lane_solid_lw)
        # dashed separators
        for i in range(1, n):
            y = (-n/2 + i) * W
            ax.hlines(y, X0, X1, color=st.lane_dash_color, lw=st.lane_dash_lw, linestyle=st.lane_dash_style)

    def _draw_compass(self, ax: plt.Axes, st: LayoutStyle):
        X0, X1 = float(self.X_min), float(self.X_max)
        Y0, Y1 = float(self.Y_min), float(self.Y_max)
        cx = 0.5 * (X0 + X1)
        cy = 0.5 * (Y0 + Y1)

        kw = dict(color=st.compass_color, fontsize=st.compass_fontsize, family=st.compass_family)
        ax.text(cx, Y1 - 1.0, "N", ha="center", va="top", **kw)
        ax.text(X1 - 1.0, cy, "E", ha="right", va="center", **kw)
        ax.text(cx, Y0 + 1.0, "S", ha="center", va="bottom", **kw)
        ax.text(X0 + 1.0, cy, "W", ha="left", va="center", **kw)
