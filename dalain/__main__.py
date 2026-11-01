from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __package__ in {None, ""}:
    from dalain.config import Config
    from dalain.optimize import mean_normal_cwef, run
else:
    from .config import Config
    from .optimize import mean_normal_cwef, run


def main():
    # Основные настройки редактируются в config.py.
    # При необходимости переопределяйте их здесь,
    # но не задавайте один параметр в двух местах без необходимости.
    config = Config(
        # Примеры переопределений:

        center_lon=19.95,
        center_lat=54.74,
        target_area_m2=100_000.0,
        cell_side_m=25.0,
        connection="attached",
        start_date="2020-01-01",
        end_date="2025-12-31",
        direction_count=72,
        point_spacing_m=50.0,
        tangent_delta_m=20.0,
        min_valid_bathymetry_fraction=0.25,
        min_finite_bathymetry_fraction=0.95,
        allow_underresolved_bathymetry=True,
        short_profile_pixel_diagonals=2.0,
    )

    run(
        config,
        objective=mean_normal_cwef,
    )


if __name__ == "__main__":
    main()