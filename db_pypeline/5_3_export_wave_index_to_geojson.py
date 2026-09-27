from __future__ import annotations

"""Шаг 10 пайплайна — экспорт WEI-индексов в GeoJSON.

Читает wave_activity.db (таблицы wave_points, wave_exposure_index,
wave_activity_summary) и записывает GeoJSON с точечным слоем.

Каждая Feature содержит:

    Геометрия
        Point(lon, lat) в WGS-84 (EPSG:4326)

    Базовые поля точки
        point_id, normal_x, normal_y, normal_azimuth_deg

    WEI-индексы (режим points и summary)
        mean_cwef_wm, e_storm_mjm, storm_threshold_wm,
        storm_percentile, k_dir, cv, n_days, n_storm_days,
        top3_sectors, r1, r2, r3, r4, wer

    Сводная статистика CWEF (только режим summary)
        n_active_days, median_cwef_wm, std_cwef_wm,
        p75_cwef_wm, p90_cwef_wm, p95_cwef_wm, p99_cwef_wm,
        max_cwef_wm, n_storm_days_p90, total_energy_mjm

Переменные окружения:
    WAVE_ACTIVITY_DB       — путь к wave_activity.db
    WAVE_INDEX_GEOJSON     — путь к выходному файлу
    WAVE_EXPORT_MODE       — "points" | "summary"  (по умолчанию: summary)
    WAVE_GEOJSON_INDENT    — отступ JSON (пусто = компактный, 2 = читаемый)
    WAVE_LOG_LEVEL         — уровень логирования
"""

import os
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "db_pypeline" / "data"

ACTIVITY_DB = Path(
    os.getenv(
        "WAVE_ACTIVITY_DB",
        DATA_DIR / "db" / "wave_activity.db",
    )
)

OUTPUT_GEOJSON = Path(
    os.getenv(
        "WAVE_INDEX_GEOJSON",
        DATA_DIR / "wave" / "wave_exposure_index.geojson",
    )
)

EXPORT_MODE = os.getenv(
    "WAVE_EXPORT_MODE",
    "summary",
)

_raw_indent = os.getenv("WAVE_GEOJSON_INDENT", "").strip()
GEOJSON_INDENT: int | None = (
    int(_raw_indent) if _raw_indent else None
)

LOG_LEVEL = os.getenv("WAVE_LOG_LEVEL", "INFO")


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


def main() -> None:
    configure_logging()

    source = ACTIVITY_DB.expanduser().resolve()
    output = OUTPUT_GEOJSON.expanduser().resolve()

    if EXPORT_MODE not in ("points", "summary"):
        raise ValueError(
            f"Unknown WAVE_EXPORT_MODE={EXPORT_MODE!r}. "
            "Use 'points' or 'summary'."
        )

    if not source.is_file():
        raise FileNotFoundError(
            f"Wave activity database not found: {source}.\n"
            "Run db_pypeline/5_2_wave_exposure_to_db.py first."
        )

    logger.info("Wave activity DB : {}", source)
    logger.info("Output GeoJSON   : {}", output)
    logger.info("Export mode      : {}", EXPORT_MODE)
    logger.info(
        "JSON indent      : {}",
        GEOJSON_INDENT if GEOJSON_INDENT is not None else "compact",
    )

    from src.waves.export import WaveIndexGeoJSONExporter
    from src.waves.storage import WaveActivityRepository

    with WaveActivityRepository(source) as repo:
        exporter = WaveIndexGeoJSONExporter(repo)
        stats = exporter.export(
            output_path=output,
            mode=EXPORT_MODE,
            indent=GEOJSON_INDENT,
        )

    logger.success(
        "Done: {} features → {}",
        stats.feature_count,
        stats.output_path,
    )
    logger.info(
        "WER range: [{:.3f} … {:.3f}], mean = {:.3f}",
        stats.min_wer if stats.min_wer is not None else 0.0,
        stats.max_wer if stats.max_wer is not None else 0.0,
        stats.mean_wer if stats.mean_wer is not None else 0.0,
    )


if __name__ == "__main__":
    main()
