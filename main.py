from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.coastline.domain import CoastlineDataset
from src.coastline.exporters import GeoJsonPointExporter
from src.coastline.point_strategies import EqualRadiusStrategy, PointSource
from src.coastline.services import CoastlinePointExtractor
from src.weather_history.GribWeatherLayerWrapper import GribWeatherLayerWrapper


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    main_coastline_path = "data/main_coastline.geojson"
    other_lines_path = "data/other_lines.geojson"
    grib_path = "data/cop_cds_cropp.grib"
    # grib_path = "data/cop_cds.grib"

    radius_points_geojson_path = output_dir / "novoross_main_radius_points.geojson"

    coastline_weather_geojson_path = output_dir / "coastline_points_with_weather.geojson"
    coastline_weather_gpkg_path = output_dir / "coastline_points_with_weather.gpkg"
    coastline_weather_per_point_dir = output_dir / "coastline_weather_points_rows"

    assignment_strategy = "idw"   # "nearest" | "idw"
    idw_power = 2.0
    idw_k = 4

    dataset = CoastlineDataset.from_geojson(
        main_path=main_coastline_path,
        other_path=other_lines_path,
        name="novoross_coastline",
    )
    dataset.print_summary()

    main_gdf = dataset.main_gdf
    if main_gdf.crs is None:
        raise ValueError("dataset.main_gdf has no CRS")

    working_crs = main_gdf.estimate_utm_crs()
    if working_crs is None:
        raise ValueError("Failed to estimate working metric CRS from main coastline")

    working_crs_str = str(working_crs)
    logger.info(f"Working metric CRS: {working_crs_str}")

    extractor = CoastlinePointExtractor()

    radius_points = extractor.extract(
        dataset=dataset,
        strategy=EqualRadiusStrategy(
            radius_step_m=500.0,
            source=PointSource.MAIN_ONLY,
            include_origin=True,
            include_endpoint=True,
            working_crs=working_crs_str,
        ),
        name="novoross_main_radius_points",
    )

    radius_points.print_summary()

    radius_points.export(
        strategy=GeoJsonPointExporter(),
        output_path=str(radius_points_geojson_path),
    )
    logger.success(f"Coastline radius points saved: {radius_points_geojson_path}")

    logger.info(f"Loading GRIB weather layer: {grib_path}")

    try:
        weather_wrapper = GribWeatherLayerWrapper.from_grib(grib_path)
    except Exception as exc:
        logger.warning(f"Direct GRIB open failed: {exc}")
        logger.info("Trying fallback via cfgrib.open_datasets(...)")
        weather_wrapper = GribWeatherLayerWrapper.from_grib_datasets(grib_path)

    logger.info(f"Weather rows loaded: {len(weather_wrapper.weather_gdf)}")
    logger.info(f"Weather columns: {list(weather_wrapper.weather_gdf.columns)}")

    assigned_gdf = weather_wrapper.assign_to_points(
        coastline_points_path=radius_points_geojson_path,
        strategy=assignment_strategy,
        output_geojson_path=coastline_weather_geojson_path,
        output_gpkg_path=coastline_weather_gpkg_path,
        output_layer_name="coastline_weather_points",
        idw_power=idw_power,
        idw_k=idw_k,
        working_crs=working_crs_str,
        aggregate_numeric=True,
    )

    logger.info(f"Assigned strategy: {assignment_strategy}")
    logger.info(f"Assigned coastline points count: {len(assigned_gdf)}")
    logger.success(f"Assignment GeoJSON saved: {coastline_weather_geojson_path}")
    logger.success(f"Assignment GPKG saved: {coastline_weather_gpkg_path}")

    exported_files = weather_wrapper.export_point_files(
        assigned_gdf=assigned_gdf,
        output_dir=coastline_weather_per_point_dir,
        coast_id_column="point_id",
        driver="GeoJSON",
    )

    logger.info(f"Per-point files count: {len(exported_files)}")
    logger.success(f"Per-point directory saved: {coastline_weather_per_point_dir}")
    logger.success("GRIB pipeline completed successfully.")


if __name__ == "__main__":
    main()
