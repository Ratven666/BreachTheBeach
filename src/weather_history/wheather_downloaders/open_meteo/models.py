from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class GridPoint:
    lat: float
    lon: float
    ring_y: int = 0
    ring_x: int = 0

    @property
    def lat_label(self) -> str:
        return f"{self.lat:.3f}"

    @property
    def lon_label(self) -> str:
        return f"{self.lon:.3f}"


@dataclass(frozen=True)
class WeatherRequest:
    geojson_path: Path
    start_date: str
    end_date: str
    daily_variables: tuple[str, ...]


@dataclass(frozen=True)
class CacheSegment:
    point: GridPoint
    start_date: str
    end_date: str
    variables_key: str
    json_path: Path
    metadata_path: Path
