from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WeatherDownloadConfig:
    cache_dir: Path = Path("data/cache/open_meteo")
    output_geojson_path: Path = Path("output/weather_daily_grid.geojson")
    model: str = "era5"
    grid_step: float = 0.25
    grid_center_offset: float = 0.125
    cover_points_with_cells: bool = True
    extra_border_cells: int = 1
    timezone: str = "GMT"
    cell_selection: str = "nearest"
    batch_size: int = 25
    request_pause_seconds: float = 0.0
    user_agent: str = "BreachTheBeach/0.1.0"
    archive_min_date: str = "1940-01-01"
    archive_lag_days: int = 7
    daily_variables: tuple[str, ...] = field(
        default_factory=lambda: (
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
        )
    )
