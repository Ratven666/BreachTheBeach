from __future__ import annotations

from abc import ABC, abstractmethod

from src.base.BBox import BBox
from src.bathymetry.domain.models import BathymetryGrid


class BathymetryLoader(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self, bbox: BBox) -> BathymetryGrid:
        raise NotImplementedError
