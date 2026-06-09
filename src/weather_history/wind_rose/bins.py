from __future__ import annotations

import numpy as np


DEFAULT_DIRECTION_LABELS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DEFAULT_DIRECTION_LABELS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]


def build_direction_edges(nsector: int) -> np.ndarray:
    if nsector <= 0:
        raise ValueError("nsector must be > 0")
    return np.linspace(0.0, 360.0, nsector + 1)


def build_direction_centers(direction_edges: np.ndarray) -> np.ndarray:
    return (direction_edges[:-1] + direction_edges[1:]) / 2.0


def build_direction_labels(nsector: int) -> list[str]:
    if nsector == 8:
        return DEFAULT_DIRECTION_LABELS_8.copy()
    if nsector == 16:
        return DEFAULT_DIRECTION_LABELS_16.copy()
    return [f"{int(v)}°" for v in build_direction_centers(build_direction_edges(nsector))]


def build_speed_bins(
    speed_values: np.ndarray,
    bins: int | list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    finite_speed = speed_values[np.isfinite(speed_values)]
    if finite_speed.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    if bins is None:
        vmax = float(np.nanmax(finite_speed))
        if vmax <= 0:
            return np.array([0.0, 1.0], dtype=float)
        return np.linspace(0.0, vmax, 7)

    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("bins integer must be > 0")
        vmax = float(np.nanmax(finite_speed))
        if vmax <= 0:
            vmax = 1.0
        return np.linspace(0.0, vmax, bins + 1)

    arr = np.asarray(bins, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("bins must contain at least two edges")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("bins must be strictly increasing")
    return arr
