from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

    def __str__(self) -> str:
        return f"({self.minx}, {self.miny}) - ({self.maxx}, {self.maxy})"


@dataclass
class CoastlineSummary:
    name: str
    crs: str | None
    bbox: BoundingBox
    main_feature_count: int
    other_feature_count: int
    total_feature_count: int
    main_total_length: float | None
    other_total_length: float | None
    total_length: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "crs": self.crs,
            "bbox": self.bbox.as_tuple(),
            "main_feature_count": self.main_feature_count,
            "other_feature_count": self.other_feature_count,
            "total_feature_count": self.total_feature_count,
            "main_total_length": self.main_total_length,
            "other_total_length": self.other_total_length,
            "total_length": self.total_length,
        }
