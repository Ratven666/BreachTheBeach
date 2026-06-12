from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.waves.services import WaveClimateBatchProcessor


def main() -> None:
    output_dir = Path("../nvrsk_calc/waves")
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = Path("../nvrsk_calc/nvrsk_equal_radius_200m_points_with_normals.geojson")
    fetch_csv_path = Path("../nvrsk_calc/fetch/fetch_daily_input.csv")
    weather_csv_path = Path("../nvrsk_calc/weather/point_weather_daily_long.csv")

    daily_output_path = output_dir / "wave_climate_daily.geojson"
    summary_output_path = output_dir / "wave_climate_summary.geojson"

    processor = WaveClimateBatchProcessor(
        overwater_factor=1.1,
        breaking_coeff=0.55,
        rho_water=1025.0,
        g=9.81,
    )

    daily_path, summary_path = processor.export(
        points_path=points_path,
        fetch_csv_path=fetch_csv_path,
        weather_csv_path=weather_csv_path,
        daily_output_path=daily_output_path,
        summary_output_path=summary_output_path,
        normal_field="normal_azimuth_deg",
    )

    logger.success(f"Daily wave climate saved to: {daily_path}")
    logger.success(f"Summary wave climate saved to: {summary_path}")


if __name__ == "__main__":
    main()
