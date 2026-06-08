from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class CoastlineExportStrategy(ABC):
    @abstractmethod
    def export(self, dataset: "CoastlineDataset", output_path: str | Path) -> Path:
        raise NotImplementedError
