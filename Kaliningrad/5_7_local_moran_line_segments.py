from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"

POINT_EXPORT_GEOJSON = DATA_DIR / "wave" / "local_moran.geojson"
COASTLINE_GEOJSON = DATA_DIR / "coastline" / "T_kaliningrad_merged.geojson"
OUTPUT_GEOJSON = DATA_DIR / "wave" / "local_moran_line_segments.geojson"

# Если None, CRS будет оценена автоматически по береговой линии
METRIC_CRS: str | None = None

LOG_LEVEL = "INFO"


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )

    from src.waves.local_moran import build_moran_line_segments

    logger.info("Point export: {}", POINT_EXPORT_GEOJSON.resolve())
    logger.info("Coastline: {}", COASTLINE_GEOJSON.resolve())
    logger.info("Output line segments: {}", OUTPUT_GEOJSON.resolve())
    logger.info("metric_crs={}", METRIC_CRS)

    stats = build_moran_line_segments(
        POINT_EXPORT_GEOJSON,
        COASTLINE_GEOJSON,
        OUTPUT_GEOJSON,
        metric_crs=METRIC_CRS,
    )

    logger.success(
        "Done: features={}, output={}, coastline_crs={}, metric_crs={}",
        stats.feature_count,
        stats.output_path,
        stats.coastline_crs,
        stats.metric_crs,
    )


if __name__ == "__main__":
    main()