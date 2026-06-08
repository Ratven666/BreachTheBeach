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
    # 2. Сервис извлечения точек
    # ---------------------------------------------------------
    extractor = CoastlinePointExtractor()

    # ---------------------------------------------------------
    # 3. Узловые точки только главной линии
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
    # 4. Точки через равный шаг только по главной линии
    # ---------------------------------------------------------
    step_points = extractor.extract(
        dataset=dataset,
        strategy=EqualStepAlongLineStrategy(
            step=0.001,  # шаг в единицах CRS
            source=PointSource.MAIN_ONLY,
            include_endpoints=True,
        ),
        name="novoross_main_step_points",
    )

    step_points.print_summary()

    step_points.export(
        strategy=GeoPackagePointExporter(layer_name="main_equal_step"),
        output_path="output/novoross_main_points.gpkg",
    )

    # ---------------------------------------------------------
    # 5. Точки по радиусу от начала только по главной линии
    # ---------------------------------------------------------
    radius_points = extractor.extract(
        dataset=dataset,
        strategy=EqualRadiusStrategy(
            radius_step=0.002,
            source=PointSource.MAIN_ONLY,
            include_origin=True,
            include_endpoint=True,
        ),
        name="novoross_main_radius_points",
    )

    radius_points.print_summary()

    radius_points.export(
        strategy=GeoPackagePointExporter(layer_name="main_equal_radius"),
        output_path="output/novoross_main_points.gpkg",
    )

    # ---------------------------------------------------------
    # 6. Комбинированный набор только для главной линии
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
                    step=0.001,
                    source=PointSource.MAIN_ONLY,
                    include_endpoints=True,
                ),
                EqualRadiusStrategy(
                    radius_step=0.0015,
                    source=PointSource.MAIN_ONLY,
                    include_origin=True,
                    include_endpoint=True,
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