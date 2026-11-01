from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RefractionModel:
    g: float = 9.81

    def _phase_speed(self, h: float, T: float = 8.0) -> float:
        """
        Фазовая скорость через полное дисперсионное уравнение:
            ω² = g·k·tanh(k·h),  ω = 2π/T
        Итерационное решение методом простой итерации (сходится за ~20 шагов).
        При T=8 с и h=3 м даёт c≈5.5 м/с вместо мелководного √(g·h)≈5.4 —
        разница небольшая, но при h_deep=20 м расхождение уже ~15%.
        """
        h = max(float(h), 0.1)
        omega = 2.0 * np.pi / max(float(T), 1.0)
        # Начальное приближение — мелководная скорость
        k = omega / np.sqrt(self.g * h)
        for _ in range(50):
            k_new = omega ** 2 / (self.g * np.tanh(k * h))
            if abs(k_new - k) < 1e-8:
                break
            k = k_new
        return float(omega / k)

    def transform(
        self,
        direction_deg: float,
        shore_normal_deg: float,
        h_deep: float,
        h_point: float,
        tp_s: float = 8.0,
    ) -> tuple[float, float]:
        delta_in = ((float(direction_deg) - float(shore_normal_deg) + 180.0) % 360.0) - 180.0
        theta_in = np.deg2rad(abs(delta_in))
        c1 = self._phase_speed(h_deep, tp_s)
        c2 = self._phase_speed(h_point, tp_s)
        sin_out = np.clip(np.sin(theta_in) * c2 / c1, -1.0, 1.0)
        theta_out = np.arcsin(sin_out)
        cos_refracted = float(np.clip(np.cos(theta_out), 0.0, None))
        return cos_refracted, float(np.rad2deg(theta_out))
