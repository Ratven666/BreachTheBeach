from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"

MODE = "calculate"
MODE = "export"

INDEX_DB = DATA_DIR / "db" / "wave_activity_indices.db"
COASTLINE_GEOJSON = DATA_DIR / "coastline" / "T_kaliningrad_merged.geojson"

RADIUS_M = 10000.0
WINDOW_WIDTH_M = 30000.0

ALONGSHORE_MODE = "abs"
METRIC_CRS: str | None = None

MAX_SNAP_DISTANCE_M: float | None = 100.0
STRICT_SNAP = False

WEIGHT_MODE = "binary"
PERMUTATIONS = 999
RANDOM_SEED = 42

# Порог по raw pseudo p-value для поля cluster
# ALPHA_RAW = 0.10
ALPHA_RAW = 0.999

# Порог по FDR-corrected q-value для поля cluster_fdr
# ALPHA_FDR = 0.10
ALPHA_FDR = 0.999

EXPORT_RUN_ID: int | None = None
OUTPUT_GEOJSON = DATA_DIR / "wave" / "local_moran.geojson"

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

    from src.waves.local_moran.service import LocalMoranService
    from src.waves.local_moran.export import export_local_moran_geojson

    mode = str(MODE).strip().lower()

    if mode == "calculate":
        logger.info("Index database: {}", INDEX_DB.resolve())
        logger.info("Coastline GeoJSON: {}", COASTLINE_GEOJSON.resolve())
        logger.info(
            "Parameters: radius_m={}, window_width_m={}, alongshore_mode={}, "
            "metric_crs={}, max_snap_distance_m={}, strict_snap={}, "
            "weight_mode={}, permutations={}, seed={}, alpha_raw={}, alpha_fdr={}",
            RADIUS_M,
            WINDOW_WIDTH_M,
            ALONGSHORE_MODE,
            METRIC_CRS,
            MAX_SNAP_DISTANCE_M,
            STRICT_SNAP,
            WEIGHT_MODE,
            PERMUTATIONS,
            RANDOM_SEED,
            ALPHA_RAW,
            ALPHA_FDR,
        )

        stats = LocalMoranService(
            INDEX_DB,
            COASTLINE_GEOJSON,
            radius_m=RADIUS_M,
            window_width_m=WINDOW_WIDTH_M,
            alongshore_mode=ALONGSHORE_MODE,
            metric_crs=METRIC_CRS,
            max_snap_distance_m=MAX_SNAP_DISTANCE_M,
            strict_snap=STRICT_SNAP,
            weight_mode=WEIGHT_MODE,
            permutations=PERMUTATIONS,
            seed=RANDOM_SEED,
            alpha_raw=ALPHA_RAW,
            alpha_fdr=ALPHA_FDR,
        ).run()

        logger.success(
            "Completed: run_id={}, total_points={}, snap_ok={}, snap_bad={}, results={}, db={}",
            stats.run_id,
            stats.n_points_total,
            stats.n_points_snap_ok,
            stats.n_points_snap_bad,
            stats.n_results,
            stats.output_db,
        )
        return

    if mode == "export":
        logger.info("Index database: {}", INDEX_DB.resolve())
        logger.info("Output GeoJSON: {}", OUTPUT_GEOJSON.resolve())
        logger.info("Export run_id: {}", EXPORT_RUN_ID)

        stats = export_local_moran_geojson(
            INDEX_DB,
            OUTPUT_GEOJSON,
            run_id=EXPORT_RUN_ID,
        )

        logger.success(
            "Exported: run_id={}, features={}, output={}",
            stats.run_id,
            stats.feature_count,
            stats.output_path,
        )
        return

    raise ValueError(f"Unsupported MODE={MODE!r}. Use 'calculate' or 'export'.")


if __name__ == "__main__":
    main()