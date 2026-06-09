from __future__ import annotations

import numpy as np

from src.weather_history.wind_rose.bins import (
    build_direction_centers,
    build_direction_edges,
    build_direction_labels,
    build_speed_bins,
)
from src.weather_history.wind_rose.WindRose import WindRose, WindRoseTable


class WindRoseBuilder:
    def __init__(
        self,
        nsector: int = 16,
        bins: int | list[float] | tuple[float, ...] | np.ndarray | None = None,
        calm_limit: float | None = None,
    ) -> None:
        self.nsector = int(nsector)
        self.bins = bins
        self.calm_limit = calm_limit

    def build(
        self,
        speed: list[float | None] | tuple[float | None, ...] | np.ndarray,
        direction: list[float | None] | tuple[float | None, ...] | np.ndarray,
        ws_unit: str | None = None,
        title: str | None = None,
    ) -> WindRose:
        speed_arr = np.asarray(speed, dtype=float)
        dir_arr = np.asarray(direction, dtype=float)

        if speed_arr.shape != dir_arr.shape:
            raise ValueError("speed and direction must have the same shape")

        mask = np.isfinite(speed_arr) & np.isfinite(dir_arr)
        speed_arr = speed_arr[mask]
        dir_arr = dir_arr[mask] % 360.0

        if self.calm_limit is not None:
            mask = speed_arr > float(self.calm_limit)
            speed_arr = speed_arr[mask]
            dir_arr = dir_arr[mask]

        direction_edges = build_direction_edges(self.nsector)
        direction_centers = build_direction_centers(direction_edges)
        direction_labels = build_direction_labels(self.nsector)
        speed_bins = build_speed_bins(speed_arr, self.bins)

        if speed_arr.size == 0:
            table = np.zeros((len(speed_bins) - 1, self.nsector), dtype=int)
            freq = np.zeros_like(table, dtype=float)
            return WindRose(
                speed=speed_arr,
                direction=dir_arr,
                ws_unit=ws_unit,
                title=title,
                table_data=WindRoseTable(
                    speed_bins=speed_bins,
                    direction_edges=direction_edges,
                    direction_centers=direction_centers,
                    direction_labels=direction_labels,
                    table=table,
                    frequencies_percent=freq,
                    total_count=0,
                ),
            )

        dir_idx = np.digitize(dir_arr, direction_edges, right=False) - 1
        dir_idx = np.where(dir_idx == self.nsector, 0, dir_idx)

        spd_idx = np.digitize(speed_arr, speed_bins, right=False) - 1
        spd_idx = np.clip(spd_idx, 0, len(speed_bins) - 2)

        table = np.zeros((len(speed_bins) - 1, self.nsector), dtype=int)
        for s_i, d_i in zip(spd_idx, dir_idx, strict=False):
            table[int(s_i), int(d_i)] += 1

        freq = (table / float(len(speed_arr))) * 100.0

        return WindRose(
            speed=speed_arr,
            direction=dir_arr,
            ws_unit=ws_unit,
            title=title,
            table_data=WindRoseTable(
                speed_bins=speed_bins,
                direction_edges=direction_edges,
                direction_centers=direction_centers,
                direction_labels=direction_labels,
                table=table,
                frequencies_percent=freq,
                total_count=int(len(speed_arr)),
            ),
        )
