from __future__ import annotations

from pathlib import Path

import xarray as xr
from loguru import logger

from src.bathymetry.domain.models import BathymetryGrid
from src.bathymetry.exporters.base import BathymetryExportStrategy


class NetCDFBathymetryExporter(BathymetryExportStrategy):
    def export(self, grid: BathymetryGrid, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ds = xr.Dataset(
            {
                "elevation": xr.DataArray(
                    grid.z,
                    dims=["lat", "lon"],
                    attrs={
                        "units": "m",
                        "long_name": "Elevation / depth",
                        "source": grid.source,
                    },
                )
            },
            coords={
                "lat": xr.DataArray(grid.lats, dims=["lat"], attrs={"units": "degrees_north"}),
                "lon": xr.DataArray(grid.lons, dims=["lon"], attrs={"units": "degrees_east"}),
            },
            attrs={
                "Conventions": "CF-1.8",
                "source": grid.source,
                "resolution_arcsec": grid.resolution_arcsec,
            },
        )

        ds.to_netcdf(path)
        logger.info(f"Grid exported to NetCDF: {path}")
        return path
