# perception/observation.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Observation:
    t: float
    detected: bool
    source: str              # "none" | "onboard" | "v2x" | "fused"
    z: Optional[np.ndarray]  # [x, y, vx, vy] or None
    v2x_state: int           # 0/1
    fov_detected: int        # 0/1
    profile: str             # "good"/"bad"/...

class SimpleObservationModel:
    """
    No noise, no delay.
    Make observation from already-updated flags (very explicit, no hidden stepping).
    """
    def make_observation(self, t: float, target,
                         fov_detected: int,
                         v2x_state: int,
                         profile: str) -> Observation:
        # ---- sanitize types ----
        fov_detected = int(fov_detected)
        v2x_state = int(v2x_state)

        # ---- explicit source selection ----
        if v2x_state == 1 and fov_detected == 1:
            source = "fused"
            detected = True
        elif v2x_state == 1 and fov_detected == 0:
            source = "v2x"
            detected = True
        elif v2x_state == 0 and fov_detected == 1:
            source = "onboard"
            detected = True
        else:
            source = "none"
            detected = False

        # ---- measurement: true state (for now) ----
        if detected:
            vx = float(getattr(target, "v_x", 0.0))
            vy = float(getattr(target, "v_y", 0.0))
            z = np.array([float(target.x), float(target.y), vx, vy], dtype=float)
        else:
            z = None

        return Observation(
            t=float(t),
            detected=detected,
            source=source,
            z=z,
            v2x_state=v2x_state,
            fov_detected=fov_detected,
            profile=str(profile),
        )
