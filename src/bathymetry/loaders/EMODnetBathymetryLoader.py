from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode

import numpy as np
import requests
from loguru import logger

from src.base.BBox import BBox
from src.bathymetry.domain.models import BathymetryGrid
from src.bathymetry.errors import BathymetryLoadError
from src.bathymetry.loaders.BathymetryLoader import BathymetryLoader


class EMODnetBathymetryLoader(BathymetryLoader):
    """
    Загрузчик батиметрии из EMODnet Bathymetry через WCS GetCoverage.

    Источник:
    - endpoint: https://ows.emodnet-bathymetry.eu/wcs
    - service: WCS 2.0.1
    - coverage: emodnet:mean

    EMODnet публикует bathymetry как WCS service, а coverage `emodnet:mean`
    присутствует в capabilities как mean depth [web:123][web:134].
    """

    def __init__(
        self,
        *,
        base_url: str = "https://ows.emodnet-bathymetry.eu/wcs",
        coverage_id: str = "emodnet:mean",
        output_dir: str | Path | None = None,
        timeout: int = 300,
        save_download: bool = False,
    ) -> None:
        self._base_url = base_url
        self._coverage_id = coverage_id
        self._output_dir = Path(output_dir) if output_dir else None
        self._timeout = timeout
        self._save_download = save_download
        self._log = logger.bind(cls=self.__class__.__name__)

        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def source_name(self) -> str:
        return "EMODnet_WCS"

    def load(self, bbox: BBox) -> BathymetryGrid:
        tif_path = self._download_geotiff(bbox)
        try:
            grid = self._read_geotiff_as_grid(tif_path)
            return grid
        finally:
            if not self._save_download and tif_path.exists():
                tif_path.unlink(missing_ok=True)

    def _download_geotiff(self, bbox: BBox) -> Path:
        params = [
            ("SERVICE", "WCS"),
            ("REQUEST", "GetCoverage"),
            ("VERSION", "2.0.1"),
            ("COVERAGEID", self._coverage_id),
            ("FORMAT", "image/tiff"),
            ("SUBSET", f"Long({bbox.west},{bbox.east})"),
            ("SUBSET", f"Lat({bbox.south},{bbox.north})"),
        ]

        request_url = f"{self._base_url}?{urlencode(params, doseq=True)}"
        self._log.info(f"Requesting EMODnet WCS coverage: {request_url}")

        try:
            response = requests.get(self._base_url, params=params, stream=True, timeout=self._timeout)
            response.raise_for_status()
        except Exception as e:
            raise BathymetryLoadError(f"Failed to download EMODnet bathymetry: {e}") from e

        if self._output_dir:
            tif_path = self._output_dir / self._build_filename(bbox)
        else:
            tmp = NamedTemporaryFile(suffix=".tif", delete=False)
            tif_path = Path(tmp.name)
            tmp.close()

        part_path = tif_path.with_suffix(tif_path.suffix + ".part")

        try:
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.replace(part_path, tif_path)
        except Exception as e:
            part_path.unlink(missing_ok=True)
            raise BathymetryLoadError(f"Failed to save EMODnet GeoTIFF: {e}") from e

        self._log.info(f"EMODnet GeoTIFF saved to: {tif_path}")
        return tif_path

    def _read_geotiff_as_grid(self, tif_path: Path) -> BathymetryGrid:
        try:
            import rasterio
        except ImportError as e:
            raise BathymetryLoadError(
                "rasterio is required to read EMODnet GeoTIFF. Install it with `poetry add rasterio`."
            ) from e

        try:
            with rasterio.open(tif_path) as src:
                z = src.read(1).astype(np.float64)
                nodata = src.nodata
                bounds = src.bounds
                width = src.width
                height = src.height

                if nodata is not None:
                    z[z == nodata] = np.nan

                lons = np.linspace(bounds.left, bounds.right, width)
                lats_desc = np.linspace(bounds.top, bounds.bottom, height)

                lats = lats_desc[::-1]
                z = np.flipud(z)

                return BathymetryGrid(
                    lats=lats,
                    lons=lons,
                    z=z,
                    source=self.source_name,
                )
        except Exception as e:
            raise BathymetryLoadError(f"Failed to read EMODnet GeoTIFF {tif_path}: {e}") from e

    @staticmethod
    def _build_filename(bbox: BBox) -> str:
        return (
            "emodnet_"
            f"{bbox.south:.4f}_{bbox.west:.4f}_{bbox.north:.4f}_{bbox.east:.4f}.tif"
        ).replace("-", "m")
