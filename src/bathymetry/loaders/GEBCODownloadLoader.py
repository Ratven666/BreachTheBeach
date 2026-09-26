from __future__ import annotations

import io

import numpy as np
import requests
import xarray as xr
from loguru import logger

from src.base.BBox import BBox
from src.bathymetry.domain.models import BathymetryGrid
from src.bathymetry.loaders.BathymetryLoader import BathymetryLoader

# REST-endpoint beta-загрузчика GEBCO (download.gebco.net)
# Возвращает NetCDF-файл по заданному bbox без аутентификации.
_GEBCO_DOWNLOAD_URL = "https://download.gebco.net/api/datasets/2025_00/subset"


class GEBCODownloadLoader(BathymetryLoader):
    """
    Загрузчик батиметрии GEBCO через HTTP-download API (download.gebco.net).

    Метод загрузки: POST-запрос с параметрами bbox, формат — netCDF.
    Подходит при недоступности OPeNDAP-сервера CEDA.

    Параметры
    ----------
    timeout : int
        Таймаут HTTP-запроса в секундах.
    """

    _FORMAT = "netCDF"

    def __init__(self, timeout: int = 120) -> None:
        self._timeout = timeout
        self._log = logger.bind(cls=self.__class__.__name__)

    @property
    def source_name(self) -> str:
        return "GEBCO_Download"

    def load(self, bbox: BBox) -> BathymetryGrid:
        self._log.info(f"Downloading GEBCO subset: bbox={bbox}")
        raw = self._fetch(bbox)
        grid = self._parse(raw)
        self._log.info(f"Downloaded grid shape={grid.shape}")
        return grid

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch(self, bbox: BBox) -> bytes:
        params = {
            "south": bbox.south,
            "north": bbox.north,
            "west": bbox.west,
            "east": bbox.east,
            "format": self._FORMAT,
        }
        response = requests.get(
            _GEBCO_DOWNLOAD_URL, params=params, timeout=self._timeout
        )
        response.raise_for_status()
        return response.content

    @staticmethod
    def _parse(raw: bytes) -> BathymetryGrid:
        ds = xr.open_dataset(io.BytesIO(raw), engine="scipy")
        try:
            lats = ds["lat"].values.astype(np.float64)
            lons = ds["lon"].values.astype(np.float64)
            z = ds["elevation"].values.astype(np.float64)
        finally:
            ds.close()
        return BathymetryGrid(lats=lats, lons=lons, z=z, source="GEBCO_Download")
