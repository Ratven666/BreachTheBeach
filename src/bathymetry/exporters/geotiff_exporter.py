from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.transform import from_bounds

from src.bathymetry.domain.models import BathymetryGrid
from src.bathymetry.exporters.base import BathymetryExportStrategy


class GeoTIFFBathymetryExporter(BathymetryExportStrategy):
    def __init__(
        self,
        *,
        crs: str = "EPSG:4326",
        compress: str = "deflate",
        nodata: float = -9999.0,
    ) -> None:
        self._crs = crs
        self._compress = compress
        self._nodata = nodata

    def export(self, grid: BathymetryGrid, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._write_geotiff(
            grid=grid,
            output_path=path,
            data=grid.z,
        )
        logger.info(f"Grid exported to GeoTIFF: {path}")
        return path

    def export_split(
        self,
        grid: BathymetryGrid,
        output_dir: str | Path,
        base_name: str = "bathymetry",
    ) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sea_path = output_dir / f"{base_name}_sea.tif"
        land_path = output_dir / f"{base_name}_land.tif"

        sea_data = np.where(grid.z < 0, grid.z, np.nan)
        land_data = np.where(grid.z >= 0, grid.z, np.nan)

        self._write_geotiff(
            grid=grid,
            output_path=sea_path,
            data=sea_data,
        )
        self._write_geotiff(
            grid=grid,
            output_path=land_path,
            data=land_data,
        )

        logger.info(f"Sea GeoTIFF exported to: {sea_path}")
        logger.info(f"Land GeoTIFF exported to: {land_path}")

        return sea_path, land_path

    def _write_geotiff(
        self,
        grid: BathymetryGrid,
        output_path: Path,
        data: np.ndarray,
    ) -> None:
        height, width = data.shape

        transform = from_bounds(
            west=grid.west,
            south=grid.south,
            east=grid.east,
            north=grid.north,
            width=width,
            height=height,
        )

        data = np.asarray(data, dtype=np.float32)

        # BathymetryGrid хранится south -> north.
        # GeoTIFF ожидает, что первая строка растра соответствует north/top.
        data = np.flipud(data)

        valid_mask = ~np.isnan(data)
        data_to_write = np.where(valid_mask, data, self._nodata).astype(np.float32)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=self._crs,
            transform=transform,
            nodata=self._nodata,
            compress=self._compress,
        ) as dst:
            dst.write(data_to_write, 1)
            dst.write_mask((valid_mask * 255).astype(np.uint8))
            dst.update_tags(
                source=grid.source,
                resolution_arcsec=str(grid.resolution_arcsec),
            )
