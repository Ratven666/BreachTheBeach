from __future__ import annotations

"""
Конвертация всех NetCDF-тайлов архива батиметрии в GeoTIFF.

Использует LocalNetCDFBathymetryLoader и GeoTIFFBathymetryExporter
из src/bathymetry — никакого прямого xarray/rasterio вне проекта.

Переменные окружения:

    BATHY_MANIFEST
        Путь к manifest.json.
        По умолчанию data/bathymetry/manifest.json.

    BATHY_GEOTIFF_DIR
        Каталог для GeoTIFF.
        По умолчанию <parent(manifest)>/geotiff/.

    BATHY_GEOTIFF_MODE
        full   — один растр со всеми значениями (по умолчанию).
        sea    — только подводные ячейки (z < 0), суша → nodata.
        land   — только надводные ячейки (z >= 0), море → nodata.
        split  — два файла: <tile>_sea.tif и <tile>_land.tif.

    BATHY_GEOTIFF_COMPRESS
        Метод сжатия rasterio. По умолчанию deflate.

    BATHY_GEOTIFF_NODATA
        Значение nodata. По умолчанию -9999.0.

    BATHY_FORCE
        1/true/yes/on — перезаписать существующие GeoTIFF.
"""

import os
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    import json

    from src.base.BBox import BBox
    from src.bathymetry.exporters.GeoTIFFBathymetryExporter import (
        GeoTIFFBathymetryExporter,
    )
    from src.bathymetry.loaders.LocalNetCDFBathymetryLoader import (
        LocalNetCDFBathymetryLoader,
    )

    manifest_path = Path(
        os.getenv("BATHY_MANIFEST", "data/bathymetry/manifest.json")
    ).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found: {manifest_path}\n"
            "Run db_pypeline/4_1_download_bathymetry.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bathymetry manifest schema.")

    root_dir = manifest_path.parent

    geotiff_dir = Path(
        os.getenv("BATHY_GEOTIFF_DIR", str(root_dir / "geotiff"))
    ).resolve()

    mode = os.getenv("BATHY_GEOTIFF_MODE", "full").lower()

    if mode not in {"full", "sea", "land", "split"}:
        raise ValueError(
            "BATHY_GEOTIFF_MODE must be full, sea, land, or split"
        )

    compress = os.getenv("BATHY_GEOTIFF_COMPRESS", "deflate")
    nodata = float(os.getenv("BATHY_GEOTIFF_NODATA", "-9999.0"))
    force = _bool_env("BATHY_FORCE")

    geotiff_dir.mkdir(parents=True, exist_ok=True)

    exporter = GeoTIFFBathymetryExporter(
        crs="EPSG:4326",
        compress=compress,
        nodata=nodata,
    )

    tiles = manifest["tiles"]

    logger.info(
        f"Converting {len(tiles)} tile(s) → {geotiff_dir}  "
        f"[mode={mode}, compress={compress}]"
    )

    produced: list[Path] = []

    for tile in tiles:
        tile_id: str = tile["tile_id"]
        nc_path = (root_dir / tile["file"]).resolve()

        if not nc_path.exists():
            raise FileNotFoundError(
                f"NetCDF tile not found: {nc_path}"
            )

        gb = tile["grid_bbox"]
        bbox = BBox(
            south=float(gb["south"]),
            west=float(gb["west"]),
            north=float(gb["north"]),
            east=float(gb["east"]),
        )

        # ── загружаем грид через проектный загрузчик ──────────────
        loader = LocalNetCDFBathymetryLoader(nc_path)
        grid = loader.load(bbox)

        # ── экспортируем в выбранном режиме ───────────────────────
        if mode == "split":
            target_sea = geotiff_dir / f"{tile_id}_sea.tif"
            target_land = geotiff_dir / f"{tile_id}_land.tif"

            if not force and target_sea.exists() and target_land.exists():
                logger.info(f"  {tile_id}: already exists, skipping")
                produced += [target_sea, target_land]
                continue

            sea_path, land_path = exporter.export_split(
                grid,
                output_dir=geotiff_dir,
                base_name=tile_id,
            )
            produced += [sea_path, land_path]

        else:
            target = geotiff_dir / f"{tile_id}.tif"

            if not force and target.exists():
                logger.info(f"  {tile_id}: already exists, skipping")
                produced.append(target)
                continue

            if mode == "sea":
                import numpy as np

                grid_sea = grid.__class__(
                    lats=grid.lats,
                    lons=grid.lons,
                    z=np.where(grid.z < 0, grid.z, float("nan")),
                    source=grid.source,
                    resolution_arcsec=grid.resolution_arcsec,
                )
                exporter.export(grid_sea, target)

            elif mode == "land":
                import numpy as np

                grid_land = grid.__class__(
                    lats=grid.lats,
                    lons=grid.lons,
                    z=np.where(grid.z >= 0, grid.z, float("nan")),
                    source=grid.source,
                    resolution_arcsec=grid.resolution_arcsec,
                )
                exporter.export(grid_land, target)

            else:  # full
                exporter.export(grid, target)

            produced.append(target)

    logger.success(
        f"Done: {len(produced)} GeoTIFF file(s) written to {geotiff_dir}"
    )

    for p in produced:
        logger.info(f"  {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()