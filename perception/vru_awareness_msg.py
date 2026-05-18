import numpy as np
from typing import Optional, Dict, Tuple

class VRU_AWR:
    """
    2-state Markov model of V2X link quality.
    State: 1 = link present, 0 = no link.
    Profiles set P(1→1)=p_keep and P(0→1)=p_recover.
    """
    DEFAULT_PROFILES: Dict[str, Tuple[float, float]] = {
        "no":      (0.0, 0.0),  # never keeps, never recovers
        "bad":     (0.3, 0.2),  # often drops, rarely recovers
        "good":    (0.9, 0.7),  # usually keeps, often recovers
        "perfect": (1.0, 1.0),  # always linked
    }

    def __init__(
        self,
        profile: str = "good",
        init_state: int = 1,
        seed: Optional[int] = None,
        profiles: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.state = int(init_state)
        self.state_traj = [self.state]
        self.profiles = dict(self.DEFAULT_PROFILES if profiles is None else profiles)
        self.set_profile(profile)

    def set_profile(self, profile: str):
        if profile not in self.profiles:
            raise ValueError(f"profile must be one of {list(self.profiles)}")
        p_keep, p_recover = self.profiles[profile]
        self.T = np.array([
            [1 - p_recover, p_recover],  # from 0 (no link)
            [1 - p_keep,    p_keep   ],  # from 1 (link)
        ], dtype=float)
        self.profile = profile

    def set_custom(self, p_keep: float, p_recover: float, name: str = "custom"):
        if not (0.0 <= p_keep <= 1.0 and 0.0 <= p_recover <= 1.0):
            raise ValueError("p_keep and p_recover must be in [0, 1].")
        self.profiles[name] = (p_keep, p_recover)
        self.set_profile(name)

    def link_check(self) -> int:
        """Advance one step; return new state as int (0/1)."""
        self.state = int(self.rng.choice((0, 1), p=self.T[self.state]))
        self.state_traj.append(self.state)
        return self.state

    def reset(self, init_state: int = 1):
        self.state = int(init_state)
        self.state_traj = [self.state]

    @property
    def as_bool(self) -> bool:
        return bool(self.state)