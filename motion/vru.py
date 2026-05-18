# ------------------------------------------------------------------
# Parameter dictionaries (tweak anytime)
# ------------------------------------------------------------------
PointMass_params = {
    "pedestrian": {
        "m"     : 80.0,      # kg   (person only)
        "R"     : 0.35,      # m    (head-to‐shoulder envelope)
        "v_max" : 1.8,       # m/s  ≈ 4 mph normal walking
        "a_max" : 1.0        # m/s² quick start / stop
    },
    "escooter": {
        "m"     : 100.0,     # kg   (20 kg scooter + 80 kg rider)
        "R"     : 0.50,      # m
        "v_max" : 9.0,      # m/s  ≈ 20 mph
        "a_max" : 3.0        # m/s² 0–25 km/h in ≈3 s
    },
    "motorcyclist": {
        "m"     : 240.0,     # kg   (160 kg bike + 80 kg rider)
        "R"     : 1.00,      # m
        "v_max" : 15.0,      # m/s  ≈ 108 km/h   (20 m/s = 108 km/h)
        "a_max" : 5.0        # m/s² modest motorcycle launch
    }
}

# ------------------------------------------------------------------
# Generic point-mass VRU model
# ------------------------------------------------------------------
import math

class PointMassNewton:
    """
    Generic point-mass kinematic model for any VRU:
        pedestrian, e-scooter rider, motorcyclist …
    """
    def __init__(self, params: dict,
                 position, velocity,
                 dt, t0=0.0,
                 vru_category="undefined",
                 rider_type="Normal"):
        # store category
        self.category = vru_category      # "pedestrian" | "escooter" | "motorcyclist"

        # physical parameters
        self.m      = params["m"]
        self.R      = params["R"]
        self.v_max  = params["v_max"]
        self.a_max  = params["a_max"]

        # state
        self.x, self.y   = position
        self.v_x, self.v_y = velocity
        self.dt          = dt
        self.t           = t0

        # rider style (optional)
        self.rider_type  = rider_type     # "Aggressive" / "Conservative" …

        # trajectory logs
        self.x_traj, self.y_traj = [], []
        self.v_x_traj, self.v_y_traj = [], []
        self.v_traj  = []                # speed
        self.action_traj = []            # applied forces
        self.t_traj  = []

    # ------------------------------------------------------------------
    def update(self, F_x, F_y):
        """Euler-integrate one time-step with force saturation."""
        # Saturate total force to +- m*a_max
        F_total = math.hypot(F_x, F_y)
        if F_total > self.m * self.a_max:
            scale = (self.m * self.a_max) / F_total
            F_x *= scale; F_y *= scale

        # 1-step explicit Euler
        self.x   += self.dt * self.v_x + 0.5 * (F_x / self.m) * self.dt**2
        self.y   += self.dt * self.v_y + 0.5 * (F_y / self.m) * self.dt**2
        self.v_x += (F_x / self.m) * self.dt
        self.v_y += (F_y / self.m) * self.dt

        # speed cap
        v = math.hypot(self.v_x, self.v_y)
        if v > self.v_max:
            self.v_x *= self.v_max / v
            self.v_y *= self.v_max / v
            v = self.v_max

        self.t += self.dt

        # log
        self.x_traj.append(self.x);  self.y_traj.append(self.y)
        self.v_x_traj.append(self.v_x); self.v_y_traj.append(self.v_y)
        self.v_traj.append(v)
        self.action_traj.append((F_x, F_y))
        self.t_traj.append(self.t)
# ------------------------------------------------------------------
