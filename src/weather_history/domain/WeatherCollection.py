from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class WeatherCollection:
    crs: Any
    dates: np.ndarray
    speed_max: np.ndarray
    speed_mean: np.ndarray
    gust_max: np.ndarray
    direction: np.ndarray
    point_ids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    req_lat: np.ndarray
    req_lon: np.ndarray
    ws_max_unit: str | None
    ws_mean_unit: str | None
    wg_max_unit: str | None
    wd_unit: str | None
    start_date: str | None
    end_date: str | None
    metric_crs: Any
    metric_coords: np.ndarray
    tree: cKDTree

    @property
    def point_count(self) -> int:
        return int(len(self.point_ids))

    @property
    def records_count(self) -> int:
        return int(len(self.dates))