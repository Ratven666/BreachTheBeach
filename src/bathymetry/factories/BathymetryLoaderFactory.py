from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.base.BBox import BBox
from src.bathymetry.loaders.BathymetryLoader import BathymetryLoader
from src.bathymetry.loaders.EMODnetBathymetryLoader import EMODnetBathymetryLoader
from src.bathymetry.loaders.gebco_opentopography_loader import GEBCOOpenTopographyLoader


@dataclass(frozen=True, slots=True)
class BathymetryCoverage:
    south: float
    west: float
    north: float
    east: float

    def contains(self, bbox: BBox) -> bool:
        return (
            self.south <= bbox.south
            and self.west <= bbox.west
            and bbox.north <= self.north
            and bbox.east <= self.east
        )


class BathymetryLoaderFactory:
    """
    Универсальная фабрика выбора загрузчика батиметрии.

    Приоритет:
    1. EMODnet — если bbox полностью лежит в зоне покрытия EMODnet.
    2. GEBCO / OpenTopography — fallback для всех остальных случаев.
    """

    EMODNET_COVERAGE = BathymetryCoverage(
        south=15.0,
        west=-36.0,
        north=90.0,
        east=43.0,
    )

    def __init__(
        self,
        *,
        emodnet_output_dir: str | Path | None = None,
        emodnet_save_download: bool = False,
        emodnet_timeout: int = 300,
        gebco_output_dir: str | Path | None = None,
        gebco_save_download: bool = False,
        gebco_timeout: int = 300,
        gebco_api_key: str | None = None,
    ) -> None:
        self._emodnet_output_dir = Path(emodnet_output_dir) if emodnet_output_dir else None
        self._emodnet_save_download = emodnet_save_download
        self._emodnet_timeout = emodnet_timeout

        self._gebco_output_dir = Path(gebco_output_dir) if gebco_output_dir else None
        self._gebco_save_download = gebco_save_download
        self._gebco_timeout = gebco_timeout
        self._gebco_api_key = gebco_api_key

    def create(self, bbox: BBox) -> BathymetryLoader:
        if self._is_emodnet_bbox(bbox):
            return EMODnetBathymetryLoader(
                output_dir=self._emodnet_output_dir,
                save_download=self._emodnet_save_download,
                timeout=self._emodnet_timeout,
            )

        return GEBCOOpenTopographyLoader(
            api_key=self._gebco_api_key,
            output_dir=self._gebco_output_dir,
            save_download=self._gebco_save_download,
            timeout=self._gebco_timeout,
        )

    def source_name_for_bbox(self, bbox: BBox) -> str:
        if self._is_emodnet_bbox(bbox):
            return "EMODnet_WCS"
        return "GEBCO_OpenTopography"

    def _is_emodnet_bbox(self, bbox: BBox) -> bool:
        return self.EMODNET_COVERAGE.contains(bbox)
