from __future__ import annotations

from loguru import logger

from src.wind_fetch.SequentialMultiDirectionFetchCalculator import (
    SequentialMultiDirectionFetchCalculator,
)
from src.wind_fetch.WindFetchConfig import WindFetchConfig
from src.wind_fetch.models import WindFetchPaths


def main() -> None:
    paths = WindFetchPaths(
        main_coastline_path="../nvrsk_calc/merged_dataset.geojson",
        other_coastline_path="../nvrsk_calc/nvrsk_other_lines.geojson",
        points_with_normals_path="../nvrsk_calc/nvrsk_equal_radius_200m_points_with_normals.geojson",
    )

    config = WindFetchConfig(
        default_offset_m=0,
        default_fetch_m=100_000.0,
        coastal_exclusion_m=0,
        normal_azimuth_field="normal_azimuth_deg",
        use_make_valid=True,
        precision_grid_m=0.05,
        azimuths_deg=list(range(0, 360)),
    )

    calculator = SequentialMultiDirectionFetchCalculator(
        paths=paths,
        config=config,
    )

    logger.info("Starting sequential multi-direction fetch calculation")
    logger.info(f"Input points: {paths.points_with_normals_path}")
    logger.info(f"Azimuth count: {len(config.azimuths_deg)}")
    logger.info(f"Offset: {config.default_offset_m} m")
    logger.info(f"Max fetch: {config.default_fetch_m} m")

    results = calculator.calculate(
        offset_m=0.1,
        show_progress=True,
        log_every_points=25,
    )

    calculator.save_combined(results, output_dir="../nvrsk_calc/fetch")
    # calculator.save_split_by_direction(results, output_dir="nvrsk_calc")

    logger.success("Fetch calculation completed successfully")


if __name__ == "__main__":
    main()