from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from .WindFetchConfig import WindFetchConfig
from .WindFetchCalculator import WindFetchCalculator
from .models import WindFetchPaths
from .WindFetchVisualizer import WindFetchVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate wind fetch length from coastline points.")
    parser.add_argument("--main-coastline", required=True, help="Path to main coastline GeoJSON")
    parser.add_argument("--other-coastline", required=False, default=None, help="Path to other coastline GeoJSON")
    parser.add_argument("--points", required=True, help="Path to GeoJSON with points and normal azimuths")
    parser.add_argument("--offset-m", type=float, default=0.1, help="Offset from coastline point along ray direction, meters")
    parser.add_argument("--default-fetch-m", type=float, default=1_000_000.0, help="Default fetch if no hit found, meters")
    parser.add_argument("--step-m", type=float, default=250.0, help="Geodesic densification step, meters")
    parser.add_argument("--output-dir", default="output", help="Directory for outputs")
    parser.add_argument("--plot", action="store_true", help="Create plot png")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = WindFetchPaths(
        main_coastline_path=Path(args.main_coastline),
        other_coastline_path=Path(args.other_coastline) if args.other_coastline else None,
        points_with_normals_path=Path(args.points),
    )
    config = WindFetchConfig(
        default_offset_m=args.offset_m,
        default_fetch_m=args.default_fetch_m,
        geodesic_step_m=args.step_m,
        output_dir=Path(args.output_dir),
    )

    calc = WindFetchCalculator(paths=paths, config=config)
    results = calc.calculate()
    saved = calc.save(results)

    logger.info(f"Saved outputs: {saved}")

    if args.plot:
        vis = WindFetchVisualizer(calc)
        plot_path = Path(args.output_dir) / "wind_fetch_map.png"
        vis.plot(results, plot_path)
        logger.info(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
