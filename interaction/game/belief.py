# interaction/game/belief.py
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Optional, Any

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class RiderTypeBelief:
    """
    Belief over rider type: P(Aggressive).
    Update uses a lightweight observation model derived from motion cues.
    Works for both straight-road (cut-in) and intersection (crossing).

    NOTE: This is intentionally simple & explicit; you can refine later.
    """
    p_aggr: float = 0.5
    # how fast belief moves (0 -> frozen, 1 -> full Bayes on each update)
    alpha: float = 1.0

    # likelihood model parameters (can be tuned per scenario)
    # "go-ness" score s in [0,1]. Higher s => more likely aggressive.
    k: float = 6.0        # sigmoid gain
    s0: float = 0.45      # center
    vlat_ref: float = 1.0 # [m/s] normalization for lateral speed

    def probs(self) -> Dict[str, float]:
        return {"Aggressive": float(self.p_aggr), "Conservative": float(1.0 - self.p_aggr)}

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def infer_go_score(
        self,
        scenario: str,
        vru_x: float, vru_y: float, vru_vx: float, vru_vy: float,
        # optional: straight-road lane-change reference
        y_start: Optional[float] = None,
        y_target: Optional[float] = None,
        # optional: intersection reference (e.g., cross direction)
        cross_axis: str = "y",
    ) -> float:
        """
        Return s in [0,1]: higher => more likely "going / committing"
        straight-road: uses lane-change progress + lateral speed
        intersection : uses |vy| (crossing) + speed toward conflict
        """
        vlat = abs(float(vru_vy)) / max(float(self.vlat_ref), 1e-6)
        vlat = _clamp(vlat, 0.0, 1.0)

        speed = math.hypot(float(vru_vx), float(vru_vy))
        speed_n = _clamp(speed / 6.0, 0.0, 1.0)  # rough normalization

        scenario = (scenario or "").lower()

        if "straight_road" in scenario:
            # lane-change progress: 0 at y_start, 1 at y_target
            if y_start is None or y_target is None:
                prog = 0.0
            else:
                denom = max(abs(float(y_target) - float(y_start)), 1e-6)
                prog = abs(float(vru_y) - float(y_start)) / denom
                prog = _clamp(prog, 0.0, 1.0)

            # cut-in is subtle: progress matters more than raw speed
            s = 0.65 * prog + 0.35 * vlat

        else:
            # intersection crossing: more obvious lateral motion and speed
            # (usually |vy| dominates)
            s = 0.60 * vlat + 0.40 * speed_n

        return _clamp(s, 0.0, 1.0)

    def update_from_motion(
        self,
        detected: bool,
        scenario: str,
        vru_x: float, vru_y: float, vru_vx: float, vru_vy: float,
        y_start: Optional[float] = None,
        y_target: Optional[float] = None,
        debug: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Bayesian-ish update using a soft observation likelihood.
        Only update if detected=True; otherwise keep prior.
        """
        if debug is not None:
            debug.clear()

        if not detected:
            if debug is not None:
                debug.update({
                    "detected": False,
                    "prior_p_aggr": float(self.p_aggr),
                    "post_p_aggr": float(self.p_aggr),
                    "note": "no update (not detected)",
                })
            return float(self.p_aggr)

        s = self.infer_go_score(
            scenario=scenario,
            vru_x=vru_x, vru_y=vru_y, vru_vx=vru_vx, vru_vy=vru_vy,
            y_start=y_start, y_target=y_target
        )

        # Convert score to "probability VRU is going/committing"
        p_go_obs = self._sigmoid(self.k * (s - self.s0))  # in (0,1)

        # Emission model: how likely is this observation under each type?
        # Aggressive tends to "go/commit" more; Conservative yields more.
        # These numbers are explicit & tunable.
        p_go_aggr = 0.75
        p_go_cons = 0.35

        L_aggr = p_go_obs * p_go_aggr + (1.0 - p_go_obs) * (1.0 - p_go_aggr)
        L_cons = p_go_obs * p_go_cons + (1.0 - p_go_obs) * (1.0 - p_go_cons)

        prior_aggr = float(self.p_aggr)
        prior_cons = 1.0 - prior_aggr

        post_aggr = prior_aggr * L_aggr
        post_cons = prior_cons * L_cons
        norm = max(post_aggr + post_cons, 1e-12)
        post_aggr /= norm

        # optional smoothing (alpha<1 slows updates)
        self.p_aggr = (1.0 - self.alpha) * prior_aggr + self.alpha * post_aggr
        self.p_aggr = _clamp(self.p_aggr, 0.001, 0.999)

        if debug is not None:
            debug.update({
                "detected": True,
                "scenario": str(scenario),
                "prior_p_aggr": float(prior_aggr),
                "go_score_s": float(s),
                "p_go_obs": float(p_go_obs),
                "L_aggr": float(L_aggr),
                "L_cons": float(L_cons),
                "post_p_aggr_bayes": float(post_aggr),
                "alpha": float(self.alpha),
                "post_p_aggr": float(self.p_aggr),
            })
        return float(self.p_aggr)

    def __repr__(self) -> str:  # pragma: no cover
        return f"RiderTypeBelief(p_aggr={self.p_aggr:.3f}, alpha={self.alpha:.2f})"
