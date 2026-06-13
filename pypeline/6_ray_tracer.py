from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.wind_fetch import (
    SequentialMultiDirectionFetchCalculator,
    WindFetchConfig,
    WindFetchPaths,
)


def main() -> None:
    paths = WindFetchPaths(
        main_coastline_path="../nvrsk_calc/merged_main.geojson",
        other_coastline_path="../nvrsk_calc/merged_other.geojson",
        points_with_normals_path="../nvrsk_calc/nvrsk_equal_radius_200m_points_with_normals.geojson",
    )

    config = WindFetchConfig(
        default_fetch_m=500_000,
        default_offset_m=10,
        azimuths_deg=list(range(0, 360, 10)),
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
