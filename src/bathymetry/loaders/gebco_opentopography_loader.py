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
from src.bathymetry.errors import (
    BathymetryConfigurationError,
    BathymetryDataReadError,
    BathymetryInvalidApiKeyError,
    BathymetryMissingApiKeyError,
    BathymetryNetworkError,
    BathymetryProviderResponseError,
)
from src.bathymetry.loaders.BathymetryLoader import BathymetryLoader


class GEBCOOpenTopographyLoader(BathymetryLoader):
    _PROVIDER_NAME = "OpenTopography"
    _API_KEY_ENV = "OPENTOPOGRAPHY_API_KEY"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://portal.opentopography.org/API/globaldem",
        demtype: str = "GEBCOIceTopo",
        output_dir: str | Path | None = None,
        timeout: int = 300,
        save_download: bool = False,
        require_api_key: bool = True,
    ) -> None:
        self._api_key = api_key or os.getenv(self._API_KEY_ENV)
        self._base_url = base_url
        self._demtype = demtype
        self._output_dir = Path(output_dir) if output_dir else None
        self._timeout = timeout
        self._save_download = save_download
        self._require_api_key = require_api_key
        self._log = logger.bind(cls=self.__class__.__name__)

        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)

        if not self._base_url:
            raise BathymetryConfigurationError("OpenTopography base_url must not be empty.")

    @property
    def source_name(self) -> str:
        return "GEBCO_OpenTopography"

    def load(self, bbox: BBox) -> BathymetryGrid:
        tif_path = self._download_geotiff(bbox)
        try:
            return self._read_geotiff_as_grid(tif_path)
        finally:
            if not self._save_download and tif_path.exists():
                tif_path.unlink(missing_ok=True)

    def _download_geotiff(self, bbox: BBox) -> Path:
        if self._require_api_key and not self._api_key:
            raise BathymetryMissingApiKeyError(
                provider=self._PROVIDER_NAME,
                variable_name=self._API_KEY_ENV,
            )

        params = {
            "demtype": self._demtype,
            "south": bbox.south,
            "north": bbox.north,
            "west": bbox.west,
            "east": bbox.east,
            "outputFormat": "GTiff",
        }

        if self._api_key:
            params["API_Key"] = self._api_key

        request_url = f"{self._base_url}?{urlencode(params)}"
        self._log.info(f"Requesting GEBCO from OpenTopography: {request_url}")

        try:
            response = requests.get(
                self._base_url,
                params=params,
                stream=True,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as e:
            raise BathymetryNetworkError(
                f"{self._PROVIDER_NAME} request timed out after {self._timeout} seconds."
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise BathymetryNetworkError(
                f"Could not connect to {self._PROVIDER_NAME}."
            ) from e
        except requests.exceptions.RequestException as e:
            raise BathymetryNetworkError(
                f"{self._PROVIDER_NAME} request failed: {e}"
            ) from e

        if response.status_code == 401:
            details = self._extract_error_preview(response)
            raise BathymetryInvalidApiKeyError(
                provider=self._PROVIDER_NAME,
                details=details or "Invalid or expired API key.",
            )

        if response.status_code == 403:
            details = self._extract_error_preview(response)
            raise BathymetryInvalidApiKeyError(
                provider=self._PROVIDER_NAME,
                details=details or "Access forbidden for the provided API key.",
            )

        if response.status_code != 200:
            details = self._extract_error_preview(response)
            raise BathymetryProviderResponseError(
                provider=self._PROVIDER_NAME,
                status_code=response.status_code,
                details=details,
            )

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
        except OSError as e:
            part_path.unlink(missing_ok=True)
            raise BathymetryDataReadError(
                f"Failed to save GeoTIFF downloaded from {self._PROVIDER_NAME}: {e}"
            ) from e

        self._log.info(f"OpenTopography GeoTIFF saved to: {tif_path}")
        return tif_path

    def _read_geotiff_as_grid(self, tif_path: Path) -> BathymetryGrid:
        try:
            import rasterio
        except ImportError as e:
            raise BathymetryConfigurationError(
                "rasterio is required to read OpenTopography GeoTIFF. "
                "Install it with `poetry add rasterio`."
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
            raise BathymetryDataReadError(
                f"Failed to read GeoTIFF from {self._PROVIDER_NAME}: {tif_path}. {e}"
            ) from e

    @staticmethod
    def _extract_error_preview(response: requests.Response, limit: int = 300) -> str:
        try:
            text = response.text.strip()
        except Exception:
            return ""
        return text[:limit]

    def _build_filename(self, bbox: BBox) -> str:
        safe_demtype = self._demtype.replace(":", "_")
        return (
            f"{safe_demtype}_"
            f"{bbox.south:.4f}_{bbox.west:.4f}_{bbox.north:.4f}_{bbox.east:.4f}.tif"
        ).replace("-", "m")
