from __future__ import annotations

from pathlib import Path

from loguru import logger

from wind_fetch import WindFetchConfig
from wind_fetch.SequentialMultiDirectionFetchCalculator import SequentialMultiDirectionFetchCalculator
from wind_fetch.models import WindFetchPaths


def main() -> None:
    paths = WindFetchPaths(
        main_coastline_path="../nvrsk_calc/for_example/merged_main.geojson",
        other_coastline_path="../nvrsk_calc/for_example/merged_other.geojson",
        points_with_normals_path="../nvrsk_calc/nvrsk_equal_radius_1000m_points_with_normals.geojson",
    )

    config = WindFetchConfig(
        default_fetch_m=100_000,
        default_offset_m=10,
        azimuths_deg=list(range(0, 360)),
        output_dir="../nvrsk_calc/fetch",
    )

    calculator = SequentialMultiDirectionFetchCalculator(
        paths=paths,
        config=config,
    )

    results = calculator.calculate()

    logger.info(f"Total fetch results: {len(results)}")

    # save_minimal: только fetch_by_point.csv + fetch_by_point.geojson
    # Для полного набора отладочных слоёв используйте save_combined()
    saved = calculator.save_minimal(results, output_dir="../nvrsk_calc/fetch")
    for key, path in saved.items():
        logger.success(f"  {key}: {path}")


if __name__ == "__main__":
    main()
