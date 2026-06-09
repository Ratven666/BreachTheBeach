from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PointExportStrategy(ABC):
    @abstractmethod
    def export(self, point_set: "CoastlinePointSet", output_path: str | Path) -> Path:
        raise NotImplementedError
