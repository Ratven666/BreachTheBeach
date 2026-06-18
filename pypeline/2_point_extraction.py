from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.coastline.domain.CoastlineDataset import CoastlineDataset
from src.coastline.point_strategies import NodePointsStrategy, EqualStepAlongLineStrategy
from src.coastline.point_strategies.EqualRadiusStrategy import EqualRadiusStrategy
from src.coastline.point_strategies.PointExtractionStrategy import PointSource
from src.coastline.services import CoastlinePointExtractor


def build_dataset(
    main_path: str,
    other_path: str,
    dataset_name: str = "nvrsk_coastline",
) -> CoastlineDataset:
    return CoastlineDataset.from_geojson(
        main_path=main_path,
        other_path=other_path,
        name=dataset_name,
    )


def main() -> None:
    coastline_path = "../nvrsk_calc/nvrsk_main_coastline.geojson"
    other_lines_path = "../nvrsk_calc/nvrsk_other_lines.geojson"
    points_output_path = Path("../nvrsk_calc/nvrsk_equal_radius_1000m_points.geojson")
    # points_output_path = Path("../nvrsk_calc/for_example/nvrsk_points_nodes_endpoints.geojson")
    # points_output_path = Path("nvrsk_calc/nvrsk_points_equal_step_200m.geojson")

    dataset = build_dataset(
        main_path=coastline_path,
        other_path=other_lines_path,
        dataset_name="nvrsk_coastline",
    )

    logger.info(f"Dataset created: {dataset.name}")
    logger.info(f"Main features: {len(dataset.main_gdf)}")
    logger.info(f"Other features: {len(dataset.other_gdf)}")
    logger.info(f"Combined features: {len(dataset.combined_gdf)}")
    logger.info(f"CRS: {dataset.crs}")

    extractor = CoastlinePointExtractor()

    point_set = extractor.extract(
        dataset=dataset,
        strategy=EqualRadiusStrategy(
            radius_step_m=200.0,
            source=PointSource.MAIN_ONLY,
            include_origin=True,
            include_endpoint=True,
            working_crs=None,
            input_crs="EPSG:4326",
        ),
        name="nvrsk_points_equal_radius_1000m",
    )

    # point_set = extractor.extract(
    #     dataset=dataset,
    #     strategy=NodePointsStrategy(
    #         source=PointSource.MAIN_ONLY,
    #         endpoints_only=False,
    #         unique=True,
    #         precision=8,
    #     ),
    #     name="nvrsk_points_nodes_endpoints",
    # )

    # point_set = extractor.extract(
    #     dataset=dataset,
    #     strategy=EqualStepAlongLineStrategy(
    #         step_m=200.0,
    #         source=PointSource.MAIN_ONLY,
    #         include_endpoints=True,
    #         working_crs=None,
    #         input_crs="EPSG:4326",
    #     ),
    #     name="nvrsk_points_equal_step_200m",
    # )


    points_output_path.parent.mkdir(parents=True, exist_ok=True)
    point_set.gdf.to_file(points_output_path, driver="GeoJSON")

    logger.success(f"Points saved to: {points_output_path}")
    logger.info(f"Extracted points: {len(point_set.gdf)}")
    print(point_set.meta)
    print(point_set.gdf.head())


if __name__ == "__main__":
    main()
