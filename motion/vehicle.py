#Class of vehicle motion model
# Current type(s) of motion: Bicycle model

import numpy as np
import math

# Bicycle model
BicycleModel_params = {
    # WB = lr + lf
    'WB': 2.5,  # wheelbase, dist from rear wheel to front wheel
    'BACKTOWHEEL': 1.0,  # dist from backend to rear wheel
    'WIDTH': 2.0,  # width of the vehicle contour
    'LENGTH': 4.5,  # length of the vehicle contour
    'Lr': 0.5,  # dist from rear wheel to (defined) center
    'Lf': 2.0,  # dist from front wheel to (defined) center
    'v_max': 20,  # maximum speed = target speed
    'v_min':0,
    'a_max':6,
    'a_min':-6,
    'M': 2000,  # kg, weight of the vehicle, Volvo XC90 weighs 1993 kg
    'alpha': 100,  # friction
}


class BicycleModel:

    def __init__(self, params, x, y, psi, v, t0, beta, yaw):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.yaw = yaw

        self.dt = 0.1
        self.t = t0

        self.beta = beta

        self.WB = params['WB']
        self.LENGTH = params['LENGTH']  # [m]
        self.BACKTOWHEEL = params['BACKTOWHEEL']
        self.Lr = params['Lr']
        self.Lf = params['Lf']
        self.WIDTH = params['WIDTH']  # width of the vehicle contour
        self.C2R = self.Lr + self.BACKTOWHEEL
        self.C2F = self.LENGTH - self.C2R

        self.alpha = params['alpha']
        self.M = params['M']

        # state constraint
        self.v_max = params['v_max']
        self.v_min = params['v_min']

        self.a_max = params['a_max']
        self.a_min = params['a_min']

        self.u_last = 0.0

        # transition matrix todo: NEED TO MODIFY FOR BICYCLE MODEL!!!
        self.A = np.array([[1, self.dt], [0, 1-self.alpha*self.dt/self.M]])
        self.B = np.array([[0], [self.dt]])

        # CCMT x1 state
        self.v_state = None
        self.acc_state = None

        # record
        self.x_traj = []
        self.y_traj = []
        self.v_hist = []
        self.yaw_hist = []
        self.t_traj = []
        self.v_state_traj=[]
        self.acc_state_traj = []
        self.acc_traj = []


    def update_state(self, a, delta):
        if a >= 0:
            a_input = min(a,self.a_max)
        else:
            a_input = max(a,self.a_min)

        a_received = a_input

        delta_reveived = delta

        self.beta = math.atan((self.Lr / (self.Lr + self.Lf)) * math.tan(delta_reveived))

        # No consideration of road friction
        self.x = self.x + self.v * math.cos(self.psi + self.beta) * self.dt
        self.y = self.y + self.v * math.sin(self.psi + self.beta) * self.dt
        self.psi = self.psi + (self.v / self.Lr) * math.sin(self.beta) * self.dt
        self.v = self.v + a_received * self.dt

        #speed constraints
        self.v = max(self.v_min,self.v)
        self.v = min(self.v_max, self.v)

        # if len(self.v_hist)>1 and self.v_hist[-1]<0.5 and self.v >= v_turn:
        #     self.v = v_turn
        # else:
        #     self.v = self.v

        self.t = self.t + self.dt
        self.yaw = self.beta + self.psi


        self.v_state = math.ceil(self.v)

        if a_received >= 0:
            self.acc_state = math.ceil(a_received)
        else:
            self.acc_state = math.floor(a_received)

        # record trajectory
        self.x_traj.append(self.x)
        self.y_traj.append(self.y)
        self.v_hist.append(self.v)
        self.t_traj.append(self.t)
        self.yaw_hist.append(self.yaw)
        self.v_state_traj.append(self.v_state)
        self.acc_state_traj.append(self.acc_state)

        self.u_last = a_received
        self.acc_traj.append(a_received)

PurePursuitControl_params = {
    'k': 0.85,  # Look forward gain     {0.85}
    'Lfc': 1.0,  # Look ahead distance
    'Kp': 0.5,  # Speed propotional gain
    'L': 2.5  # Wheelbase of vehicle
}

# Control algorithm: Pure pursuit control
class PurePursuitControl:
    def __init__(self, params):
        # only keep params; model, cx, cy will be passed into methods
        self.k   = params['k']
        self.Lfc = params['Lfc']
        self.Kp  = params['Kp']
        self.L   = params['L']     # wheelbase

    def calc_target_index(self, model, cx, cy):
        """
        Compute the look-ahead waypoint index using cumulative-distance + binary search.
        Returns an integer index into cx, cy.
        """
        # 1) look-ahead distance
        Lf = self.k * model.v + self.Lfc

        # 2) cumulative arc-length along path
        cx_arr = np.asarray(cx)
        cy_arr = np.asarray(cy)
        ds = np.hypot(np.diff(cx_arr), np.diff(cy_arr))
        s  = np.concatenate([[0.0], np.cumsum(ds)])

        # 3) find projected path-coordinate s0 of the vehicle
        px, py = model.x, model.y
        # nearest waypoint
        d2 = (cx_arr - px)**2 + (cy_arr - py)**2
        i_wp = int(np.argmin(d2))
        best_s0 = s[i_wp]
        best_d2 = d2[i_wp]
        # refine by projecting onto the two adjacent segments
        for j in (i_wp-1, i_wp):
            if 0 <= j < len(cx_arr)-1:
                vx, vy = cx_arr[j+1]-cx_arr[j], cy_arr[j+1]-cy_arr[j]
                wx, wy = px-cx_arr[j],       py-cy_arr[j]
                t = (wx*vx + wy*vy) / (vx*vx + vy*vy)
                t = max(0.0, min(1.0, t))
                proj_s = s[j] + t*ds[j]
                proj_x = cx_arr[j] + t*vx
                proj_y = cy_arr[j] + t*vy
                dist2  = (proj_x-px)**2 + (proj_y-py)**2
                if dist2 < best_d2:
                    best_d2 = dist2
                    best_s0 = proj_s

        # 4) target path-coordinate
        s_target = best_s0 + Lf

        # 5) binary‐search for the smallest index with s[i] ≥ s_target
        i_target = int(np.searchsorted(s, s_target))
        return min(i_target, len(cx_arr)-1)


    def pure_pursuit_control(self, model, cx, cy, prev_index=0):
        """
        Returns (delta, target_index).
        delta = steering command [rad]
        target_index = index of the waypoint we are tracking
        """
        # compute/look-ahead index
        ind = self.calc_target_index(model, cx, cy)
        # optional: don’t go backwards along the path
        if prev_index > ind:
            ind = prev_index

        # goal point
        tx, ty = cx[ind], cy[ind]

        # heading error in vehicle frame (assume beta already in model.psi if used)
        alpha = math.atan2(ty - model.y, tx - model.x) - model.psi

        # steering law for bicycle model
        Lf = self.k * model.v + self.Lfc
        delta = math.atan2(2.0 * self.L * math.sin(alpha) / Lf, 1.0)

        return delta, ind