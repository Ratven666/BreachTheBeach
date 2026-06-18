from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.bathymetry.domain.models import BathymetryGrid, BathymetryProfile


class BathymetryExportStrategy(ABC):
    @abstractmethod
    def export(self, grid: BathymetryGrid, output_path: str | Path) -> Path:
        raise NotImplementedError


class ProfileExportStrategy(ABC):
    @abstractmethod
    def export(self, profile: BathymetryProfile, output_path: str | Path) -> Path:
        raise NotImplementedError
