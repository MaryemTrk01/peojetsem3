from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FusionConfig:
    # weights must sum to 1.0 (recommended)
    w_vibration: float = 0.5
    w_audio: float = 0.4
    w_temp: float = 0.1

    # temperature -> health mapping
    t_ok_c: float = 70.0   # <= => 100
    t_bad_c: float = 100.0 # >= => 0

    # RUL mapping (simple demo, adjust with your domain)
    rul_max_days: float = 60.0
    H_low: float = 20.0
    H_high: float = 90.0


def temp_to_health(temp_c: float, cfg: FusionConfig) -> float:
    if temp_c <= cfg.t_ok_c:
        return 100.0
    if temp_c >= cfg.t_bad_c:
        return 0.0
    return 100.0 * (1.0 - (temp_c - cfg.t_ok_c) / (cfg.t_bad_c - cfg.t_ok_c))


def global_health(health_vib: float, health_audio: float, temp_c: float, cfg: FusionConfig) -> float:
    ht = temp_to_health(float(temp_c), cfg)
    H = cfg.w_vibration * float(health_vib) + cfg.w_audio * float(health_audio) + cfg.w_temp * ht
    return float(np.clip(H, 0.0, 100.0))


def rul_from_health_linear(H: float, cfg: FusionConfig) -> float:
    """
    Simple linear RUL mapping in days:
      - H <= H_low  -> 0 days
      - H >= H_high -> rul_max_days
      - else linear between.
    """
    H = float(H)
    if H <= cfg.H_low:
        return 0.0
    if H >= cfg.H_high:
        return float(cfg.rul_max_days)
    return float(cfg.rul_max_days * (H - cfg.H_low) / max(1e-12, (cfg.H_high - cfg.H_low)))
# ---- Compatibility wrapper expected by app.py ----
# If your module already has a FusionRUL class, we just alias it.
try:
    FusionRULModel = FusionRUL  # type: ignore
except NameError:
    class FusionRULModel:
        """
        Minimal wrapper class so the backend can import FusionRULModel.
        Edit defaults here if needed.
        """
        def __init__(self, w_vib: float = 0.5, w_audio: float = 0.5):
            self.w_vib = w_vib
            self.w_audio = w_audio

        def predict(self, health_vib: float, health_audio: float, temp_c: float = 25.0):
            # Simple fusion (clamped 0..100)
            health_global = self.w_vib * health_vib + self.w_audio * health_audio

            # Optional temperature penalty (simple example)
            if temp_c >= 60:
                health_global -= 15
            elif temp_c >= 45:
                health_global -= 7

            health_global = max(0.0, min(100.0, float(health_global)))

            # Simple linear RUL mapping (edit if you want)
            # health 100 -> 365 days, health 0 -> 0 days
            rul_days = max(0.0, (health_global / 100.0) * 365.0)

            return {
                "health_global": health_global,
                "rul_days": rul_days
            }
