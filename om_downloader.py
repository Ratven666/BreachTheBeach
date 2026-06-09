from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.coastline.domain import CoastlineDataset
from src.coastline.exporters import GeoJsonPointExporter
from src.coastline.point_strategies import EqualRadiusStrategy, PointSource
from src.coastline.services import CoastlinePointExtractor
from src.weather_history.wheather_downloaders.open_meteo import WeatherHistoryService, WeatherDownloadConfig


def main() -> None:
    dataset = CoastlineDataset.from_geojson(
        main_path="data/main_coastline.geojson",
        other_path="data/other_lines.geojson",
        name="novoross_coastline",
    )
    dataset.print_summary()

    main_gdf = dataset.main_gdf
    if main_gdf.crs is None:
        raise ValueError("dataset.main_gdf has no CRS")

    working_crs = main_gdf.estimate_utm_crs()
    if working_crs is None:
        raise ValueError("Failed to estimate working metric CRS from main coastline")

    logger.info(f"Working metric CRS: {working_crs}")

    extractor = CoastlinePointExtractor()

    radius_points = extractor.extract(
        dataset=dataset,
        strategy=EqualRadiusStrategy(
            radius_step_m=500.0,
            source=PointSource.MAIN_ONLY,
            include_origin=True,
            include_endpoint=True,
            working_crs=str(working_crs),
        ),
        name="novoross_main_radius_points",
    )

    radius_points.print_summary()

    radius_points_geojson_path = "output/novoross_main_radius_points.geojson"
    radius_points.export(
        strategy=GeoJsonPointExporter(),
        output_path=radius_points_geojson_path,
    )

    weather_service = WeatherHistoryService(
        config=WeatherDownloadConfig(
            cache_dir=Path("data/cache/open_meteo"),
            output_geojson_path=Path("output/weather_daily_grid.geojson"),
            model="era5",
            grid_step=0.25,
            grid_center_offset=0.125,
            cover_points_with_cells=True,
            extra_border_cells=1,
            batch_size=20,
            timezone="GMT",
            cell_selection="nearest",
        )
    )

    weather_result = weather_service.download_from_geojson(
        geojson_path=radius_points_geojson_path,
        start_date="2019-01-01",
        end_date="2025-12-31",
        output_geojson_path="output/weather_daily_grid.geojson",
    )

    logger.info(f"Source bbox: {weather_result['source_bbox']}")
    logger.info(f"Weather bbox: {weather_result['weather_bbox']}")
    logger.info(f"Weather grid points: {weather_result['grid_points_count']}")
    logger.info(f"Weather GeoJSON: {weather_result['output_geojson_path']}")
    logger.success("Main coastline point extraction completed successfully.")


if __name__ == "__main__":
    main()
