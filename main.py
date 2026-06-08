from __future__ import annotations

from loguru import logger

from src.coastline.domain import CoastlineDataset
from src.coastline.exporters import (
    GeoJsonCoastlineExporter,
    GeoJsonPointExporter,
    GeoPackageCoastlineExporter,
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
    # 2. Экспорт исходных линий
    # ---------------------------------------------------------
    dataset.export(
        strategy=GeoJsonCoastlineExporter(),
        output_path="output/novoross_coastline.geojson",
    )

    dataset.export(
        strategy=GeoPackageCoastlineExporter(
            main_layer="main_coastline",
            other_layer="other_coastline",
        ),
        output_path="output/novoross_coastline.gpkg",
    )

    # ---------------------------------------------------------
    # 3. Сервис извлечения точек
    # ---------------------------------------------------------
    extractor = CoastlinePointExtractor()

    # ---------------------------------------------------------
    # 4. Узловые точки по всем линиям
    #    endpoints_only=False -> все вершины
    # ---------------------------------------------------------
    node_points = extractor.extract(
        dataset=dataset,
        strategy=NodePointsStrategy(
            source=PointSource.ALL_LINES,
            endpoints_only=False,
            unique=True,
        ),
        name="novoross_nodes",
    )

    node_points.print_summary()

    node_points.export(
        strategy=GeoJsonPointExporter(),
        output_path="output/novoross_nodes.geojson",
    )

    node_points.export(
        strategy=GeoPackagePointExporter(layer_name="nodes"),
        output_path="output/novoross_points.gpkg",
    )

    # ---------------------------------------------------------
    # 5. Точки через равный шаг только по основной линии
    # ---------------------------------------------------------
    step_points = extractor.extract(
        dataset=dataset,
        strategy=EqualStepAlongLineStrategy(
            step=0.001,  # шаг в единицах CRS
            source=PointSource.MAIN_ONLY,
            include_endpoints=True,
        ),
        name="novoross_step_points",
    )

    step_points.print_summary()

    step_points.export(
        strategy=GeoPackagePointExporter(layer_name="equal_step"),
        output_path="output/novoross_points.gpkg",
    )

    # ---------------------------------------------------------
    # 6. Точки по радиусу от начала по всем линиям
    # ---------------------------------------------------------
    radius_points = extractor.extract(
        dataset=dataset,
        strategy=EqualRadiusStrategy(
            radius_step=0.002,
            source=PointSource.ALL_LINES,
            include_origin=True,
            include_endpoint=True,
        ),
        name="novoross_radius_points",
    )

    radius_points.print_summary()

    radius_points.export(
        strategy=GeoPackagePointExporter(layer_name="equal_radius"),
        output_path="output/novoross_points.gpkg",
    )

    # ---------------------------------------------------------
    # 7. Комбинированный набор:
    #    endpoints основной линии + шаг по основной + радиусы по всем
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
                    source=PointSource.ALL_LINES,
                    include_origin=True,
                    include_endpoint=True,
                ),
            ],
            drop_duplicates=True,
            precision=8,
        ),
        name="novoross_combined_points",
    )

    combined_points.print_summary()

    combined_points.export(
        strategy=GeoJsonPointExporter(),
        output_path="output/novoross_combined_points.geojson",
    )

    combined_points.export(
        strategy=GeoPackagePointExporter(layer_name="combined"),
        output_path="output/novoross_points.gpkg",
    )

    logger.success("All exports completed successfully.")


if __name__ == "__main__":
    main()