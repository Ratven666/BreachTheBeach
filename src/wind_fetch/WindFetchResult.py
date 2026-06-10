from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindFetchResult:
    point_id: int
    source_point_lon: float
    source_point_lat: float
    start_point_lon: float
    start_point_lat: float
    normal_azimuth_deg: float
    ray_azimuth_deg: float
    fetch_length_m: float
    hit_found: bool
    hit_lon: float | None
    hit_lat: float | None
    used_default_value: bool


@dataclass(frozen=True)
class WindFetchPaths:
    main_coastline_path: Path
    other_coastline_path: Path | None
    points_with_normals_path: Path
