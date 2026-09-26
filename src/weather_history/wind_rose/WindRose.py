from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class WindRoseTable:
    speed_bins: np.ndarray
    direction_edges: np.ndarray
    direction_centers: np.ndarray
    direction_labels: list[str]
    table: np.ndarray
    frequencies_percent: np.ndarray
    total_count: int

    @property
    def sector_count(self) -> int:
        return int(len(self.direction_centers))

    @property
    def speed_class_count(self) -> int:
        return int(len(self.speed_bins) - 1)

    def direction_frequency_percent(self) -> np.ndarray:
        return np.sum(self.frequencies_percent, axis=0)

    def as_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for i in range(self.speed_class_count):
            bin_left = float(self.speed_bins[i])
            bin_right = float(self.speed_bins[i + 1])
            for j in range(self.sector_count):
                rows.append(
                    {
                        "speed_bin_left": bin_left,
                        "speed_bin_right": bin_right,
                        "direction_center_deg": float(self.direction_centers[j]),
                        "direction_label": self.direction_labels[j],
                        "count": int(self.table[i, j]),
                        "frequency_percent": float(self.frequencies_percent[i, j]),
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class WindRose:
    speed: np.ndarray
    direction: np.ndarray
    table_data: WindRoseTable
    ws_unit: str | None = None
    title: str | None = None

    @property
    def calm_count(self) -> int:
        return int(np.sum(np.nan_to_num(self.speed, nan=0.0) <= 0.0))

    @property
    def sample_count(self) -> int:
        return int(len(self.speed))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "wind_speed": self.speed,
                "wind_direction": self.direction,
            }
        )
