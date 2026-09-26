from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from src.base.BBox import BBox
from src.bathymetry.domain.models import BathymetryGrid
from src.bathymetry.errors import BathymetryLoadError
from src.bathymetry.loaders.BathymetryLoader import BathymetryLoader


class LocalNetCDFBathymetryLoader(BathymetryLoader):
    def __init__(
        self,
        file_path: str | Path,
        *,
        lat_name: str = "lat",
        lon_name: str = "lon",
        elevation_name: str = "elevation",
        source_name: str = "LocalNetCDF",
    ) -> None:
        self._raw_file_path = Path(file_path)
        self._lat_name = lat_name
        self._lon_name = lon_name
        self._elevation_name = elevation_name
        self._source_name = source_name
        self._log = logger.bind(cls=self.__class__.__name__)

    @property
    def source_name(self) -> str:
        return self._source_name

    def load(self, bbox: BBox) -> BathymetryGrid:
        file_path = self._resolve_path(self._raw_file_path)

        if not file_path.exists():
            raise BathymetryLoadError(
                "NetCDF file not found.\n"
                f"  requested: {self._raw_file_path}\n"
                f"  resolved : {file_path}\n"
                f"  cwd      : {Path.cwd()}"
            )

        self._log.info(f"Loading local NetCDF bathymetry: {file_path}")

        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            raise BathymetryLoadError(
                f"Failed to open NetCDF file: {file_path}"
            ) from e

        try:
            self._validate_dataset(ds)

            subset = ds.sel(
                {
                    self._lat_name: slice(bbox.south, bbox.north),
                    self._lon_name: slice(bbox.west, bbox.east),
                }
            )

            lats = subset[self._lat_name].values.astype(np.float64)
            lons = subset[self._lon_name].values.astype(np.float64)
            z = subset[self._elevation_name].values.astype(np.float64)

            if lats.size == 0 or lons.size == 0 or z.size == 0:
                raise BathymetryLoadError(
                    f"Empty subset for bbox={bbox}. "
                    f"Check bbox bounds against dataset coverage."
                )

            return BathymetryGrid(
                lats=lats,
                lons=lons,
                z=z,
                source=self._source_name,
            )
        except BathymetryLoadError:
            raise
        except Exception as e:
            raise BathymetryLoadError(
                f"Failed to read subset from NetCDF file: {file_path}"
            ) from e
        finally:
            ds.close()

    def _validate_dataset(self, ds: xr.Dataset) -> None:
        missing = [
            name
            for name in (self._lat_name, self._lon_name, self._elevation_name)
            if name not in ds.variables and name not in ds.coords
        ]
        if missing:
            raise BathymetryLoadError(
                f"Dataset missing required variables/coords: {missing}"
            )

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
