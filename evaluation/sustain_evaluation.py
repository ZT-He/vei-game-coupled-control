# This is the class of the sustainable evaluation metric

import numpy as np
import math

class SustainEvaluator:
    def __init__(self,start_pos,finish_pos,veh,desired_veh_velo,vru):
        self.eff_index = 0.0
        self.safety_index = 0.0
        self.comfort_index = 0.0

        self.overall_index = self.eff_index + self.safety_index +self.comfort_index

        self.veh = veh
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.desired_veh_velo = desired_veh_velo
        self.vru = vru

        # record
        self.eff_index_traj = []
        self.ttc_ind_traj = []
        self.safety_index_traj = []

        self.jerk_traj = []
        self.yaw_rate_traj = []
        self.comfort_index_traj = []
        self.overall_index_traj = []

        self.has_collision = any(x == -1 for x in self.ttc_ind_traj) # flag for collision state

    # Updated efficiency calculation for curved path
    def _path_length(self, cx, cy):
        """Return cumulative arc-length of a polyline."""
        cx = np.asarray(cx);  cy = np.asarray(cy)
        seg = np.hypot(np.diff(cx), np.diff(cy))
        return float(seg.sum())

    # ------------------------------------------------------------------
    def cal_eff_index(self, actual_finish_time, ref_x, ref_y):
        """
        Efficiency index =
            1 − (actual_time − ideal_time) / ideal_time,
        where ideal_time = path_length / desired_velocity.

        Parameters
        ----------
        actual_finish_time : float   # [s] measured time to complete the run
        ref_x, ref_y       : 1-D sequences of way-points describing
                             the *reference* trajectory the vehicle should follow.
        """
        path_len = self._path_length(ref_x, ref_y)           # metres
        ideal_t  = path_len / self.desired_veh_velo          # seconds

        # print('Ideal travel time:',ideal_t)

        self.eff_index = 1.0 - (actual_finish_time - ideal_t) / ideal_t
        self.eff_index = max(min(self.eff_index, 1.0), 0.0)  # clamp to [0,1]

        # log
        self.eff_index_traj.append(self.eff_index)

    def cal_ttc_ind(self,collision_state):
        # calculate the propotion of the sum of safe zone and attention zone based on the TTC calculation
        # ttc = math.sqrt((self.veh.x - self.vru.x) ** 2 + (self.veh.y - self.vru.y) ** 2) / self.veh.v
        """
        Updated TTC based on relative distance and speed

        TTC =  distance ‖r‖  /  relative closing speed
             = ‖r‖ / max(-r̂·v_rel , 0)

        • r  =  vru_pos − veh_pos
        • v_rel = vru_vel − veh_vel
                  (veh velocity resolved with yaw+β)
        If the two agents are diverging (closing-speed ≤ 0) we treat TTC = +∞.
        """
        # --- relative position ---
        rx = self.vru.x - self.veh.x
        ry = self.vru.y - self.veh.y
        dist = math.hypot(rx, ry)

        # --- vehicle velocity vector (CG frame) ---
        v_veh_x = self.veh.v * math.cos(self.veh.yaw)
        v_veh_y = self.veh.v * math.sin(self.veh.yaw)

        # --- relative velocity (vru w.r.t. vehicle) ---
        dvx = self.vru.v_x - v_veh_x
        dvy = self.vru.v_y - v_veh_y

        # closing speed = – r̂ · v_rel
        if dist > 1e-6:
            closing_speed = -(rx * dvx + ry * dvy) / dist
        else:
            closing_speed = float('inf')  # practically on top

        # TTC
        if closing_speed > 0:
            ttc = dist / closing_speed
        else:
            ttc = float('inf')  # diverging or stationary wrt each other

        if not collision_state:
            if ttc>2:
                ttc_ind = 2 # safe zone
            elif 1< ttc <=2:
                ttc_ind = 1 # attention zone
            elif 0< ttc <= 1:
                ttc_ind = 0  # alert zone
            else:
                ttc_ind = 2 # ttc negative: meaning veh and vru are apart
        else:
            ttc_ind = -1 # collision zone

        self.ttc_ind_traj.append(ttc_ind)

    # TODO: use simple safety calculation as below
    # def cal_safety_index(self, v_impact=None, w1=0.6, w0=0.2, q=2):
    #     """
    #     Time-weighted risk with impact-speed penalty.
    #     Also sets self.has_collision for downstream zeroing of eff/comfort.
    #     """
    #     eps = 1e-6
    #     N = len(self.ttc_ind_traj)
    #     if N == 0:
    #         self.safety_index = 0.0
    #         self.has_collision = False
    #         self.safety_index_traj.append(self.safety_index)
    #         return
    #
    #     N2 = sum(1 for x in self.ttc_ind_traj if x == 2)
    #     N1 = sum(1 for x in self.ttc_ind_traj if x == 1)
    #     N0 = sum(1 for x in self.ttc_ind_traj if x == 0)
    #     Ncoll = sum(1 for x in self.ttc_ind_traj if x == -1)
    #     has_coll = (Ncoll > 0)
    #     self.has_collision = has_coll  # <-- expose collision to other metrics
    #
    #     s_occ = (1.0 * N2 + w1 * N1 + w0 * N0) / max(N, 1)
    #     s_occ = max(0.0, min(1.0, s_occ))
    #
    #     if not has_coll:
    #         self.safety_index = s_occ
    #     else:
    #         if v_impact is None:
    #             v_impact = getattr(self, "v_impact", None)
    #         if v_impact is None:
    #             v_impact = float(self.veh.v)
    #
    #         v_des = max(float(self.desired_veh_velo), eps)
    #         r = max(0.0, min(v_impact / v_des, 1.0))
    #         penalty = min(1.0, (2.0 * r) ** q)  # r>=0.5 => penalty>=1 => zero out
    #         self.safety_index = max(0.0, s_occ * (1.0 - penalty))
    #
    #     self.safety_index_traj.append(self.safety_index)
    #
    def cal_safety_index(
            self,
            v_impact=None,
            w1=0.6,
            w0=0.2,
            base_penalty=0.7,
            v_scale=10.0,
            q=2.0,
    ):
        """
        Safety index based on TTC time-occupancy score (s_occ) and collision penalty.

        1) No collision: safety = s_occ
        2) Collision: always penalize strongly (>= base_penalty), and penalize more as
           absolute impact speed increases (no normalization by desired speed).

        penalty(v) = base_penalty + (1-base_penalty) * (1 - exp(-(v/v_scale)^q))
        safety = s_occ * (1 - penalty)

        - base_penalty in [0,1): minimum punishment whenever collision happens
        - v_scale (m/s): speed scale controlling how fast penalty saturates to 1
        - q >= 1: curvature (higher => more aggressive punishment growth)
        """
        eps = 1e-6
        N = len(self.ttc_ind_traj)
        if N == 0:
            self.safety_index = 0.0
            self.has_collision = False
            self.safety_index_traj.append(self.safety_index)
            return

        # occupancy score from TTC bins
        N2 = sum(1 for x in self.ttc_ind_traj if x == 2)
        N1 = sum(1 for x in self.ttc_ind_traj if x == 1)
        N0 = sum(1 for x in self.ttc_ind_traj if x == 0)
        Ncoll = sum(1 for x in self.ttc_ind_traj if x == -1)

        has_coll = (Ncoll > 0)
        self.has_collision = has_coll

        s_occ = (1.0 * N2 + float(w1) * N1 + float(w0) * N0) / max(N, 1)
        s_occ = max(0.0, min(1.0, s_occ))

        # case 1) no collision
        if not has_coll:
            self.safety_index = s_occ
            self.safety_index_traj.append(self.safety_index)
            return

        # case 2) no collision
        self.safety_index = 0.0
        self.safety_index_traj.append(self.safety_index)

        # # case 2) collision: always penalize, more with higher absolute impact speed
        # if v_impact is None:
        #     v_impact = getattr(self, "v_impact", None)
        # if v_impact is None:
        #     v_impact = float(self.veh.v)
        #
        # v = max(0.0, float(v_impact))  # absolute impact speed
        # v_scale = max(float(v_scale), eps)
        # q = max(float(q), 1.0)
        #
        # # smooth monotonic saturation in [0,1)
        # g = 1.0 - math.exp(-((v / v_scale) ** q))  # g(0)=0, g->1 as v increases
        #
        # base_penalty = float(base_penalty)
        # base_penalty = max(0.0, min(0.999999, base_penalty))  # avoid exactly 1
        # penalty = base_penalty + (1.0 - base_penalty) * g
        # penalty = max(0.0, min(1.0, penalty))
        #
        # self.safety_index = max(0.0, s_occ * (1.0 - penalty))
        # self.safety_index_traj.append(self.safety_index)


    def cal_jerk(self, acc_traj,yaw_hist):
        dt = self.veh.dt

        for i in range(1, len(acc_traj)):
            jerk  = (acc_traj[i] - acc_traj[i-1] )/dt
            self.jerk_traj.append(jerk)

        for j in range(1, len(yaw_hist)):
            yaw_rate  = (acc_traj[j] - acc_traj[j-1] )/dt
            self.yaw_rate_traj.append(yaw_rate)


    # orignal cmft calculation only on acc_traj (lon)
    # def cal_cmft_index(self, acc_traj):
    #     # define cmft_ind 0 as discomfort and 1 as comfort
    #     self.cal_jerk(acc_traj=acc_traj)
    #
    #     #Define the comfort & discomfort condition
    #     desired_acc_max = 2
    #     desired_acc_min = -3.5
    #     desired_jerk_max = 5
    #     desired_jerk_min = -5
    #
    #     for x, y in zip(acc_traj, self.jerk_traj):
    #         if x > desired_acc_max or x < desired_acc_min or y > desired_jerk_max or y < desired_jerk_min:
    #             cmf_ind = 0 # discomfort
    #         else:
    #             cmf_ind = 1 # comfort
    #
    #         self.comfort_index_traj.append(cmf_ind)
    #
    #     total_elements = len(self.comfort_index_traj)  # Total number of elements in the list
    #     sum_of_ones = sum(self.comfort_index_traj)  # Sum of all 1s in the list
    #     if total_elements == 0:
    #         self.comfort_index = 1  # To avoid division by zero if the list is empty
    #     else:
    #         self.comfort_index = sum_of_ones / total_elements

    def cal_cmft_index(self, acc_traj, yaw_hist):
        """
        cmf_ind: 1 = comfort, 0 = discomfort
        Adds yaw-rate / yaw-acc (yaw-rate change) gating.
        """
        dt = self.veh.dt

        # reset to avoid accumulation across calls
        self.comfort_index_traj = []

        # existing jerk calculation (keeps your current logic)
        self.cal_jerk(acc_traj=acc_traj, yaw_hist=yaw_hist)

        # --- thresholds (existing) ---
        desired_acc_max = 2.0
        desired_acc_min = -3.5
        desired_jerk_max = 5.0
        desired_jerk_min = -5.0

        # --- new yaw thresholds (from literature) ---
        # Kim et al. reported onset thresholds (average) around:
        # yaw velocity ~17.88 deg/s, yaw acceleration ~14.50 deg/s^2  :contentReference[oaicite:1]{index=1}
        # yaw_rate_max = np.deg2rad(17.88)  # rad/s
        # yaw_acc_max = np.deg2rad(14.50)  # rad/s^2

        yaw_rate_max = np.deg2rad(20.0)  # rad/s
        yaw_acc_max = np.deg2rad(15.0)  # rad/s^2

        # --- compute yaw_rate and yaw_acc from yaw_hist ---
        yaw = np.asarray(yaw_hist, dtype=float)

        # unwrap to avoid 2*pi jumps (assumes yaw in radians)
        yaw = np.unwrap(yaw)

        yaw_rate = np.zeros_like(yaw)
        yaw_rate[1:] = (yaw[1:] - yaw[:-1]) / dt

        yaw_acc = np.zeros_like(yaw)
        yaw_acc[1:] = (yaw_rate[1:] - yaw_rate[:-1]) / dt

        # --- combine all comfort gates ---
        for a, j, r, ar in zip(acc_traj, self.jerk_traj, yaw_rate, yaw_acc):
            discomfort = (
                    (a > desired_acc_max) or (a < desired_acc_min) or
                    (j > desired_jerk_max) or (j < desired_jerk_min) or
                    (abs(r) > yaw_rate_max) or  # abrupt turning
                    (abs(ar) > yaw_acc_max)  # abrupt yaw-rate change
            )
            self.comfort_index_traj.append(0 if discomfort else 1)

        total = len(self.comfort_index_traj)
        self.comfort_index = 1.0 if total == 0 else (sum(self.comfort_index_traj) / total)

    def cal_overall_index(self, coe_eff, coe_safety, coe_cmft):
        # Backward-safe: infer collision flag if not set yet
        if not hasattr(self, "has_collision"):
            self.has_collision = any(x == -1 for x in self.ttc_ind_traj)

        if self.has_collision:
            # Force eff & comfort to zero regardless of their computed values when collision happens
            self.eff_index = 0.0
            self.comfort_index = 0.0

        self.overall_index = (
                coe_eff * self.eff_index
                + coe_safety * self.safety_index
                + coe_cmft * self.comfort_index
        )

        self.overall_index_traj.append(self.overall_index)