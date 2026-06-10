from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class WindFetchConfig:
    """
    Конфигурация расчёта fetch.

    Все расстояния задаются в метрах.
    Все углы — azimuth/bearing:
    0 = north, 90 = east, 180 = south, 270 = west.
    """

    output_dir: Path = Path("output")

    multi_output_csv_name: str = "multi_direction_fetch.csv"
    multi_output_points_name: str = "multi_direction_source_points.geojson"
    multi_output_start_points_name: str = "multi_direction_start_points.geojson"
    multi_output_offset_segments_name: str = "multi_direction_offset_segments.geojson"
    multi_output_rays_name: str = "multi_direction_rays.geojson"
    multi_output_hit_points_name: str = "multi_direction_hit_points.geojson"
    multi_output_split_dirname: str = "multi_direction_split"

    default_offset_m: float = 1.0
    default_fetch_m: float = 100_000.0

    coastal_exclusion_m: float = 1.0
    normal_azimuth_field: str = "normal_azimuth_deg"

    use_make_valid: bool = True
    precision_grid_m: float = 0.05

    azimuths_deg: list[float] = field(
        default_factory=lambda: [float(v) for v in range(360)]
    )
