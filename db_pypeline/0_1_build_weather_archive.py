"""
Слияние рабочих SQLite-баз с метеоданными в один компактный архив.

Режим работы задаётся константами:
- APPEND_TO_EXISTING_ARCHIVE = True  -> дозапись в существующий архив;
- OVERWRITE_ARCHIVE = True           -> полная пересборка архива с нуля.

Одновременно включать оба режима нельзя.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.weather_history.archive import (
    CompactWeatherDatabaseBuilder,
    discover_databases,
)

# ---------------------------------------------------------------------------
# Параметры сборки — редактировать здесь
# ---------------------------------------------------------------------------

SOURCE_PATHS: list[str | Path] = [
    Path("../sourse_db/Sakhalin_and_islands_whether_db/data/db"),
]

SOURCE_GLOB_PATTERN: str = "*.db"
SOURCE_RECURSIVE: bool = True

OUTPUT_PATH: Path = Path("data/db/weather_compact.db")

TARGET_MODEL: str = "era5"

APPEND_TO_EXISTING_ARCHIVE: bool = True
OVERWRITE_ARCHIVE: bool = False

CACHE_MIB: int = 256
PAGE_SIZE: int = 16_384

LOG_LEVEL: str = "DEBUG"

# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


def main() -> None:
    configure_logging()

    if APPEND_TO_EXISTING_ARCHIVE and OVERWRITE_ARCHIVE:
        raise ValueError(
            "APPEND_TO_EXISTING_ARCHIVE=True и OVERWRITE_ARCHIVE=True "
            "одновременно задавать нельзя."
        )

    sources = discover_databases(
        SOURCE_PATHS,
        pattern=SOURCE_GLOB_PATTERN,
        recursive=SOURCE_RECURSIVE,
    )

    output_resolved = OUTPUT_PATH.expanduser().resolve()
    sources = [p for p in sources if p.expanduser().resolve() != output_resolved]

    if not sources:
        logger.warning("Не найдено ни одной исходной базы — выходим.")
        return

    logger.info("Найдено источников: {}", len(sources))
    for path in sources:
        logger.debug("  {}", path)

    logger.info(
        "Режим сборки: {}",
        "append" if APPEND_TO_EXISTING_ARCHIVE else "replace",
    )
    logger.info("Итоговая БД: {}", output_resolved)
    logger.info("Модель: {}", TARGET_MODEL)

    builder = CompactWeatherDatabaseBuilder(
        output_path=OUTPUT_PATH,
        model=TARGET_MODEL,
        overwrite=OVERWRITE_ARCHIVE,
        append=APPEND_TO_EXISTING_ARCHIVE,
        cache_mib=CACHE_MIB,
        page_size=PAGE_SIZE,
    )

    sources_with_progress = tqdm(
        sources,
        total=len(sources),
        desc="Merging databases",
        unit="db",
        dynamic_ncols=True,
        mininterval=0.2,
    )

    try:
        stats = builder.build(sources_with_progress)
    finally:
        sources_with_progress.close()

    logger.success("Архив готов: {}", stats.output_path)
    logger.info("Профиль архива: {}", stats.profile)
    logger.info("Узлов сетки: {}", stats.point_count)
    logger.info("Суточных записей: {}", stats.day_count)

    added_points_total = sum(src.points_added for src in stats.sources)
    added_days_total = sum(src.days_added for src in stats.sources)
    duplicate_days_total = sum(src.duplicate_days for src in stats.sources)
    skipped_total = sum(1 for src in stats.sources if src.skipped)

    logger.info(
        "Итого по запуску: points +{}, days +{}, duplicates {}, skipped {}",
        added_points_total,
        added_days_total,
        duplicate_days_total,
        skipped_total,
    )

    for src in stats.sources:
        logger.info(
            "  {}: points +{}, days +{}, duplicates {}, skipped {}",
            src.path.name,
            src.points_added,
            src.days_added,
            src.duplicate_days,
            src.skipped,
        )


if __name__ == "__main__":
    main()