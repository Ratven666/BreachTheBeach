from src.wind_fetch.SequentialMultiDirectionFetchCalculator import (
    SequentialMultiDirectionFetchCalculator,
)
from src.wind_fetch.WindFetchConfig import WindFetchConfig
from src.wind_fetch.models import WindFetchPaths


def main() -> None:
    paths = WindFetchPaths(
        main_coastline_path="output/merged_main.geojson",
        other_coastline_path="output/merged_other.geojson",
        points_with_normals_path="output/points_with_normals.geojson",
    )

    config = WindFetchConfig(
        default_offset_m=1.0,
        default_fetch_m=100_000.0,
        coastal_exclusion_m=1.0,
        normal_azimuth_field="normal_azimuth_deg",
        use_make_valid=True,
        precision_grid_m=0.05,
        azimuths_deg=[0, 15, 30, 45, 60, 90, 120, 180, 240, 300],
    )

    calculator = SequentialMultiDirectionFetchCalculator(
        paths=paths,
        config=config,
    )

    results = calculator.calculate(
        offset_m=1.0,
        show_progress=True,
        log_every_points=25,
    )

    calculator.save_combined(results, output_dir="output")
    calculator.save_split_by_direction(results, output_dir="output")


if __name__ == "__main__":
    main()