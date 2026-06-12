from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RefractionModel:
    g: float = 9.81

    def transform(
        self,
        direction_deg: float,
        shore_normal_deg: float,
        h_deep: float,
        h_point: float,
    ) -> tuple[float, float]:
        delta_in = ((float(direction_deg) - float(shore_normal_deg) + 180.0) % 360.0) - 180.0
        theta_in = np.deg2rad(abs(delta_in))
        c1 = np.sqrt(self.g * max(float(h_deep), 0.1))
        c2 = np.sqrt(self.g * max(float(h_point), 0.1))
        sin_out = np.clip(np.sin(theta_in) * c2 / c1, -1.0, 1.0)
        theta_out = np.arcsin(sin_out)
        cos_refracted = float(np.clip(np.cos(theta_out), 0.0, None))
        return cos_refracted, float(np.rad2deg(theta_out))
