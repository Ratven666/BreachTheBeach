# src/weather_history/wind_rose/WindRoseBuilder.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindRose:
    nsector: int
    frequencies: np.ndarray   # shape (nsector,)  — доли [0..1]
    mean_speeds: np.ndarray   # shape (nsector,)
    sector_centers: np.ndarray  # азимуты центров секторов, градусы


class WindRoseBuilder:
    """
    Строит розу ветров по массивам скоростей и направлений.

    Исправление: границы секторов сдвинуты на полшага (half_step),
    чтобы северный сектор (0°/360°) не получал двойную ширину.
    """

    def __init__(self, nsector: int = 16) -> None:
        if nsector < 4 or nsector % 2 != 0:
            raise ValueError("nsector must be even and >= 4")
        self.nsector = nsector

    def build(
        self,
        speeds: np.ndarray,
        directions: np.ndarray,
    ) -> WindRose:
        speeds = np.asarray(speeds, dtype=float)
        directions = np.asarray(directions, dtype=float) % 360.0

        step = 360.0 / self.nsector
        half_step = step / 2.0

        # Границы секторов сдвинуты на -half_step, чтобы центры совпадали
        # с кардинальными направлениями: 0°, 22.5°, 45° … при nsector=16
        edges = np.linspace(-half_step, 360.0 - half_step, self.nsector + 1)

        # Нормализуем направления в диапазон [edges[0], edges[-1])
        dirs_shifted = (directions - (-half_step)) % 360.0

        dir_idx = np.digitize(dirs_shifted, edges - (-half_step)) - 1
        dir_idx = np.clip(dir_idx, 0, self.nsector - 1)

        frequencies = np.zeros(self.nsector, dtype=float)
        mean_speeds = np.zeros(self.nsector, dtype=float)

        total = len(directions)
        for s in range(self.nsector):
            mask = dir_idx == s
            count = np.sum(mask)
            frequencies[s] = count / total if total > 0 else 0.0
            mean_speeds[s] = float(np.mean(speeds[mask])) if count > 0 else 0.0

        sector_centers = np.arange(self.nsector) * step  # 0, step, 2*step …

        return WindRose(
            nsector=self.nsector,
            frequencies=frequencies,
            mean_speeds=mean_speeds,
            sector_centers=sector_centers,
        )