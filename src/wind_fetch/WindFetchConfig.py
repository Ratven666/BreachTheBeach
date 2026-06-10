from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindFetchConfig:
    default_offset_m: float = 0.1
    default_fetch_m: float = 1_000_000.0
    geodesic_step_m: float = 250.0
    max_segments_per_ray: int = 8000
    bbox_expand_m: float = 5000.0
    output_dir: Path = Path("output")
    output_geojson_name: str = "wind_fetch.geojson"
    output_csv_name: str = "wind_fetch.csv"
    output_plot_name: str = "wind_fetch_map.geojson"
