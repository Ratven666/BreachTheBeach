from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SMBWaveGrowthModel:
    g: float = 9.81

    def calculate(self, wind_speed_ms: float, fetch_m: float) -> tuple[float, float]:
        u = max(float(wind_speed_ms), 0.5)
        f = max(float(fetch_m), 1.0)
        x = self.g * f / (u**2)
        hs = 0.283 * u**2 / self.g * np.tanh(0.53 * x**0.75)
        tp = 7.54 * u / self.g * np.tanh(0.833 * x**0.375)
        return float(hs), float(tp)
