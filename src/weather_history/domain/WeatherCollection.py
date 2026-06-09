from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class WeatherCollection:
    crs: Any
    dates: np.ndarray                 # (T,)
    speed: np.ndarray                 # (N, T)
    direction: np.ndarray             # (N, T)
    point_ids: np.ndarray             # (N,)
    lat: np.ndarray                   # (N,)
    lon: np.ndarray                   # (N,)
    req_lat: np.ndarray               # (N,)
    req_lon: np.ndarray               # (N,)
    ws_unit: str | None
    wd_unit: str | None
    start_date: str | None
    end_date: str | None
    metric_crs: Any
    metric_coords: np.ndarray         # (N, 2)
    tree: cKDTree

    @property
    def point_count(self) -> int:
        return int(len(self.point_ids))

    @property
    def records_count(self) -> int:
        return int(len(self.dates))