from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WaveEnergyCalculator:
    rho_water: float = 1025.0
    g: float = 9.81

    def wave_power(self, hs_m: float, tp_s: float) -> float:
        return float(self.rho_water * self.g**2 * hs_m**2 * tp_s / (64.0 * np.pi))

    def cwef(self, wave_power_wm: float, cos_shore: float) -> float:
        return float(max(0.0, wave_power_wm * cos_shore))
