from __future__ import annotations

from loguru import logger

from src.coastline.domain import CoastlineDataset
from src.coastline.exporters import (
    GeoJsonPointExporter,
    GeoPackagePointExporter,
)
from src.coastline.point_strategies import (
    CombinedStrategy,
    EqualRadiusStrategy,
    EqualStepAlongLineStrategy,
    NodePointsStrategy,
    PointSource,
)
from src.coastline.services import CoastlinePointExtractor


def main() -> None:
    # ---------------------------------------------------------
    # 1. Загрузка береговых линий
    # ---------------------------------------------------------
    dataset = CoastlineDataset.from_geojson(
        main_path="data/main_coastline.geojson",
        other_path="data/other_lines.geojson",
        name="novoross_coastline",
    )

    dataset.print_summary()

    # ---------------------------------------------------------
    # 2. Единая рабочая метрическая CRS для всех стратегий
    # ---------------------------------------------------------
    # Вариант 1: зафиксировать вручную, если ты точно знаешь зону
    # working_crs = "EPSG:32637"

    # Вариант 2: оценить по главной линии
    main_gdf = dataset.main_gdf
    if main_gdf.crs is None:
        raise ValueError("dataset.main_gdf has no CRS")

    working_crs = main_gdf.estimate_utm_crs()
    if working_crs is None:
        raise ValueError("Failed to estimate working metric CRS from main coastline")

    logger.info(f"Working metric CRS: {working_crs}")

    # ---------------------------------------------------------
    # 3. Сервис извлечения точек
    # ---------------------------------------------------------
    extractor = CoastlinePointExtractor()

    # ---------------------------------------------------------
    # 4. Узловые точки только главной линии
    #    endpoints_only=False -> все вершины main
    # ---------------------------------------------------------
    node_points = extractor.extract(
        dataset=dataset,
        strategy=NodePointsStrategy(
            source=PointSource.MAIN_ONLY,
            endpoints_only=False,
            unique=True,
        ),
        name="novoross_main_nodes",
    )

    node_points.print_summary()

    node_points.export(
        strategy=GeoJsonPointExporter(),
        output_path="output/novoross_main_nodes.geojson",
    )

    node_points.export(
        strategy=GeoPackagePointExporter(layer_name="main_nodes"),
        output_path="output/novoross_main_points.gpkg",
    )

    # ---------------------------------------------------------
    # 5. Точки через равный шаг только по главной линии
    #    шаг теперь задаётся в МЕТРАХ
    # ---------------------------------------------------------
    step_points = extractor.extract(
        dataset=dataset,
        strategy=EqualStepAlongLineStrategy(
            step_m=150.0,
            source=PointSource.MAIN_ONLY,
            include_endpoints=True,
            working_crs=str(working_crs),
        ),
        name="novoross_main_step_points",
    )

    step_points.print_summary()

    step_points.export(
        strategy=GeoPackagePointExporter(layer_name="main_equal_step"),
        output_path="output/novoross_main_points.gpkg",
    )

    # ---------------------------------------------------------
    # 6. Точки по радиусу от начала только по главной линии
    #    радиус теперь задаётся в МЕТРАХ
    # ---------------------------------------------------------
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

    radius_points.export(
        strategy=GeoPackagePointExporter(layer_name="main_equal_radius"),
        output_path="output/novoross_main_points.gpkg",
    )

    # ---------------------------------------------------------
    # 7. Комбинированный набор только для главной линии
    #    endpoints main + равный шаг main + радиусы main
    # ---------------------------------------------------------
    combined_points = extractor.extract(
        dataset=dataset,
        strategy=CombinedStrategy(
            strategies=[
                NodePointsStrategy(
                    source=PointSource.MAIN_ONLY,
                    endpoints_only=True,
                    unique=True,
                ),
                EqualStepAlongLineStrategy(
                    step_m=150.0,
                    source=PointSource.MAIN_ONLY,
                    include_endpoints=True,
                    working_crs=str(working_crs),
                ),
                EqualRadiusStrategy(
                    radius_step_m=500.0,
                    source=PointSource.MAIN_ONLY,
                    include_origin=True,
                    include_endpoint=True,
                    working_crs=str(working_crs),
                ),
            ],
            drop_duplicates=True,
            precision=8,
        ),
        name="novoross_main_combined_points",
    )

    combined_points.print_summary()

    combined_points.export(
        strategy=GeoJsonPointExporter(),
        output_path="output/novoross_main_combined_points.geojson",
    )

    combined_points.export(
        strategy=GeoPackagePointExporter(layer_name="main_combined"),
        output_path="output/novoross_main_points.gpkg",
    )

    logger.success("Main coastline point extraction completed successfully.")


if __name__ == "__main__":
    main()