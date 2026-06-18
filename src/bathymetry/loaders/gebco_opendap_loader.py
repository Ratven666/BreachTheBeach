from __future__ import annotations

import importlib.util

import numpy as np
import xarray as xr
from loguru import logger

from src.base.BBox import BBox
from src.bathymetry.domain.models import BathymetryGrid
from src.bathymetry.errors import BathymetryLoadError
from src.bathymetry.loaders.BathymetryLoader import BathymetryLoader

_GEBCO_OPENDAP_URL = (
    "https://dap.ceda.ac.uk/thredds/dodsC/neodc/gebco/data/gebco_2025/GEBCO_2025.nc"
)


class GEBCOOpenDAPLoader(BathymetryLoader):
    """
    Загрузчик GEBCO через OPeNDAP.

    Важно: endpoint через CEDA может требовать аутентификацию,
    поэтому loader не должен считаться основным публичным источником.
    """

    def __init__(self, opendap_url: str = _GEBCO_OPENDAP_URL, engine: str | None = None) -> None:
        self._url = opendap_url
        self._engine = engine
        self._log = logger.bind(cls=self.__class__.__name__)

    @property
    def source_name(self) -> str:
        return "GEBCO_OPeNDAP"

    def load(self, bbox: BBox) -> BathymetryGrid:
        engine = self._resolve_engine()
        self._log.info(
            f"Loading GEBCO via OPeNDAP: bbox={bbox}, url={self._url}, engine={engine}"
        )

        try:
            ds = xr.open_dataset(self._url, engine=engine)
        except Exception as e:
            raise BathymetryLoadError(
                f"Failed to open GEBCO OPeNDAP dataset with engine='{engine}': {e}"
            ) from e

        try:
            subset = ds.sel(
                lat=slice(bbox.south, bbox.north),
                lon=slice(bbox.west, bbox.east),
            )

            return BathymetryGrid(
                lats=subset["lat"].values.astype(np.float64),
                lons=subset["lon"].values.astype(np.float64),
                z=subset["elevation"].values.astype(np.float64),
                source="GEBCO_OPeNDAP",
            )
        except Exception as e:
            raise BathymetryLoadError(
                f"Failed to subset GEBCO OPeNDAP dataset: {e}"
            ) from e
        finally:
            ds.close()

    def _resolve_engine(self) -> str:
        if self._engine is not None:
            return self._engine
        if importlib.util.find_spec("netCDF4") is not None:
            return "netcdf4"
        if importlib.util.find_spec("pydap") is not None:
            return "pydap"
        raise BathymetryLoadError("No available OPeNDAP backend. Install netCDF4 or pydap.")
