from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from loguru import logger

from src.coastline.domain import CoastlineDataset
from src.coastline.domain.CoastlinePointSet import CoastlinePointSet, PointSetMeta
from src.coastline.services.CoastlineNormalService import (
    CoastlineNormalConfig,
    CoastlineNormalService,
)


def load_point_set(
    points_path: str | Path,
    name: str = "nvrsk_equal_radius_200m_points",
) -> CoastlinePointSet:
    gdf = gpd.read_file(points_path)

    if gdf.empty:
        raise ValueError(f"Point file is empty: {points_path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    meta = PointSetMeta(
        name=name,
        source_dataset_name="nvrsk_coastline",
        strategy_name="EqualRadiusStrategy",
        source_mode="main_only",
        points_count=len(gdf),
    )

    return CoastlinePointSet(gdf=gdf, meta=meta)


def main() -> None:
    coastline_path   = "../nvrsk_calc/nvrsk_main_coastline.geojson"
    other_lines_path = "../nvrsk_calc/nvrsk_other_lines.geojson"
    points_path      = "../nvrsk_calc/nvrsk_equal_radius_1000m_points.geojson"

    output_dir = Path("../nvrsk_calc")
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_points_path = output_dir / "nvrsk_equal_radius_1000m_points_with_normals.geojson"
    # normal_lines не используется ни одним последующим шагом.
    # Для визуализации в QGIS раскомментируйте:
    # normal_lines_path = output_dir / "nvrsk_equal_radius_200m_normal_lines.geojson"

    dataset = CoastlineDataset.from_geojson(
        main_path=coastline_path,
        other_path=other_lines_path,
        name="nvrsk_coastline",
    )

    point_set = load_point_set(
        points_path=points_path,
        name="nvrsk_equal_radius_200m_points",
    )

    logger.info(f"Dataset created: {dataset.name}")
    logger.info(f"Main features: {len(dataset.main_gdf)}")
    logger.info(f"Other features: {len(dataset.other_gdf)}")
    logger.info(f"Loaded points: {len(point_set.gdf)}")

    normal_service = CoastlineNormalService(
        CoastlineNormalConfig(
            sea_side="right",
            normal_length_m=200.0,
            tangent_delta_m=5.0,
            working_crs=str(dataset.metric_crs),
        )
    )

    normal_points = normal_service.build_points_with_normals(
        point_set=point_set,
        dataset=dataset,
        name="nvrsk_equal_radius_200m_points_with_normals",
    )

    normal_points.to_geojson(normal_points_path)
    logger.success(f"Normal points saved to: {normal_points_path}")

    # Раскомментируйте для сохранения линий нормалей (только QGIS-визуализация):
    # normal_lines = normal_points.to_normal_lines_gdf(normal_length_m=200.0)
    # normal_lines.to_file(normal_lines_path, driver="GeoJSON")
    # logger.success(f"Normal lines saved to: {normal_lines_path}")

    print(normal_points)
    print(normal_points.summary())


if __name__ == "__main__":
    main()
