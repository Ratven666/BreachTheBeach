from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd

from secret.OPEN_TOPOGRAPHY_API import OP_TOP_KEY
from src.base.BBox import BBox
from src.bathymetry import (
    BathymetryCache,
    BathymetryLoaderFactory,
    BathymetryService,
    EMODnetBathymetryLoader,
    GEBCOOpenTopographyLoader,
    GeoTIFFBathymetryExporter,
    NetCDFBathymetryExporter,
)

LoaderMode = Literal["auto", "emodnet", "gebco"]


@dataclass(frozen=True, slots=True)
class BathymetryDownloadCase:
    name: str
    source_geojson: Path
    loader_mode: LoaderMode = "auto"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "nvrsk_calc" / "bathymetry_from_bbox"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def bbox_from_geojson(path: str | Path) -> BBox:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"GeoJSON is empty: {path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds

    return BBox(
        south=float(miny),
        west=float(minx),
        north=float(maxy),
        east=float(maxx),
    )


def build_loader(*, bbox: BBox, mode: LoaderMode, raw_dir: Path):
    if mode == "emodnet":
        return EMODnetBathymetryLoader(
            output_dir=raw_dir / "emodnet",
            save_download=True,
        )

    if mode == "gebco":
        return GEBCOOpenTopographyLoader(
            api_key=OP_TOP_KEY,
            output_dir=raw_dir / "gebco",
            save_download=True,
        )

    factory = BathymetryLoaderFactory(
        emodnet_output_dir=raw_dir / "emodnet",
        emodnet_save_download=True,
        gebco_output_dir=raw_dir / "gebco",
        gebco_save_download=True,
        gebco_api_key=OP_TOP_KEY,
    )
    return factory.create(bbox)


def build_service(case_dir: Path, bbox: BBox, mode: LoaderMode) -> BathymetryService:
    raw_dir = case_dir / "raw"
    cache_dir = case_dir / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(
        bbox=bbox,
        mode=mode,
        raw_dir=raw_dir,
    )
    print(f"Selected bathymetry source: {loader.source_name}")

    cache = BathymetryCache(cache_dir)

    return BathymetryService(
        loader=loader,
        cache=cache,
        n_profile_points=300,
        interp_method="linear",
    )


def run_case(case: BathymetryDownloadCase) -> None:
    print("=" * 100)
    print(f"Running bathymetry download for: {case.name}")
    print(f"Source geometry: {case.source_geojson}")
    print(f"Loader mode: {case.loader_mode}")

    bbox = bbox_from_geojson(case.source_geojson)
    print(
        f"BBox -> south={bbox.south:.6f}, west={bbox.west:.6f}, "
        f"north={bbox.north:.6f}, east={bbox.east:.6f}"
    )

    case_dir = OUTPUT_ROOT / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    service = build_service(
        case_dir=case_dir,
        bbox=bbox,
        mode=case.loader_mode,
    )

    geotiff_exporter = GeoTIFFBathymetryExporter()

    # 1) скачиваем батиметрию
    grid = service.fetch(bbox)
    print(f"Grid shape: {grid.shape}, depth range: [{grid.min_depth:.1f}, {grid.max_depth:.1f}] m")

    # 2) сохраняем растры
    service.export_grid(
        geotiff_exporter,
        case_dir / "bathymetry_georeferenced.tif",
    )

    service.export_grid(
        NetCDFBathymetryExporter(),
        case_dir / "bathymetry.nc",
    )

    sea_tif_path, land_tif_path = geotiff_exporter.export_split(
        grid,
        case_dir,
        base_name="bathymetry_georeferenced",
    )

    # 3) рисуем только карту глубин в PNG
    service.plot_grid(
        contour_interval=50.0,  # при необходимости поменяй шаг изообат
        output_path=case_dir / "bathymetry_map.png",
        show=False,
    )

    print("Saved files:")
    print(f"  - Full GeoTIFF : {case_dir / 'bathymetry_georeferenced.tif'}")
    print(f"  - Sea GeoTIFF  : {sea_tif_path}")
    print(f"  - Land GeoTIFF : {land_tif_path}")
    print(f"  - NetCDF       : {case_dir / 'bathymetry.nc'}")
    print(f"  - Map PNG      : {case_dir / 'bathymetry_map.png'}")
    print(f"  - Output dir   : {case_dir}")

def main() -> None:
    case = BathymetryDownloadCase(
        name="nvrsk_merged_dataset",
        source_geojson=PROJECT_ROOT / "nvrsk_calc" / "merged_dataset.geojson",
        loader_mode="auto",   # можно "emodnet" или "gebco"
    )
    run_case(case)


if __name__ == "__main__":
    main()