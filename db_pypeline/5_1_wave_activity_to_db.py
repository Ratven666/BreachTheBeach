from __future__ import annotations

"""Расчёт волновой активности и запись минимальной результирующей SQLite-БД.

Предварительно должны быть выполнены этапы 0-4 db_pypeline.
Настройки можно переопределить переменными окружения:

    WAVE_WEATHER_DB, WAVE_COASTLINE_DB, WAVE_BATHY_MANIFEST,
    WAVE_OUTPUT_DB, WAVE_NORMAL_SOURCE_ID, WAVE_START_DATE,
    WAVE_END_DATE, WAVE_OVERWATER_FACTOR, WAVE_BATHY_STEPS,
    WAVE_COMMIT_BATCH_SIZE, WAVE_OVERWRITE.
"""

import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "db_pypeline" / "data"
WEATHER_DB = Path(os.getenv("WAVE_WEATHER_DB", DATA_DIR / "db" / "weather_compact.db"))
COASTLINE_DB = Path(os.getenv("WAVE_COASTLINE_DB", DATA_DIR / "db" / "coastline.db"))
BATHY_MANIFEST = Path(
    os.getenv("WAVE_BATHY_MANIFEST", DATA_DIR / "bathymetry" / "manifest.json")
)
OUTPUT_DB = Path(os.getenv("WAVE_OUTPUT_DB", DATA_DIR / "db" / "wave_activity.db"))
NORMAL_SOURCE_ID = (
    int(os.environ["WAVE_NORMAL_SOURCE_ID"])
    if os.getenv("WAVE_NORMAL_SOURCE_ID")
    else None
)
START_DATE = os.getenv("WAVE_START_DATE") or None
END_DATE = os.getenv("WAVE_END_DATE") or None
OVERWATER_FACTOR = float(os.getenv("WAVE_OVERWATER_FACTOR", "1.1"))
BATHY_STEPS = int(os.getenv("WAVE_BATHY_STEPS", "200"))
COMMIT_BATCH_SIZE = int(os.getenv("WAVE_COMMIT_BATCH_SIZE", "5000"))
OVERWRITE = os.getenv("WAVE_OVERWRITE", "false").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("WAVE_LOG_LEVEL", "INFO")


def _validate_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
    )


def main() -> None:
    configure_logging()
    for path, label in (
        (WEATHER_DB, "weather archive"),
        (COASTLINE_DB, "coastline database"),
        (BATHY_MANIFEST, "bathymetry manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path.resolve()}")

    output = OUTPUT_DB.expanduser().resolve()
    if output.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output database already exists: {output}. "
            "Set WAVE_OVERWRITE=true to replace it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    _remove_sqlite_files(temporary)

    os.environ["COASTLINE_DATABASE_URL"] = (
        f"sqlite:///{COASTLINE_DB.expanduser().resolve().as_posix()}"
    )
    from src.bathymetry.archive import BathymetryArchive
    from src.coastline.storage.db import SessionLocal
    from src.weather_history.archive import CompactWeatherRepository
    from src.waves.services.wave_activity_db_service import (
        WaveActivityDatabaseBuilder,
        resolve_normal_source_id,
    )
    from src.waves.storage import WaveActivityRepository

    logger.info("Weather archive: {}", WEATHER_DB.resolve())
    logger.info("Coastline database: {}", COASTLINE_DB.resolve())
    logger.info("Bathymetry manifest: {}", BATHY_MANIFEST.resolve())
    logger.info("Output database: {}", output)

    try:
        with SessionLocal() as coastline_session:
            normal_source_id = resolve_normal_source_id(
                coastline_session,
                NORMAL_SOURCE_ID,
            )
            logger.info("Selected normal_source_id={}", normal_source_id)
            bathymetry = BathymetryArchive(BATHY_MANIFEST)
            with CompactWeatherRepository(WEATHER_DB) as weather_repository:
                with WaveActivityRepository(temporary) as output_repository:
                    stats = WaveActivityDatabaseBuilder(
                        coastline_session=coastline_session,
                        weather_repository=weather_repository,
                        bathymetry_archive=bathymetry,
                        output_repository=output_repository,
                        normal_source_id=normal_source_id,
                        start_date=_validate_date(START_DATE),
                        end_date=_validate_date(END_DATE),
                        overwater_factor=OVERWATER_FACTOR,
                        bathy_n_steps=BATHY_STEPS,
                        commit_batch_size=COMMIT_BATCH_SIZE,
                    ).run()
        if output.exists():
            _remove_sqlite_files(output)
        os.replace(temporary, output)
    except Exception:
        _remove_sqlite_files(temporary)
        logger.exception("Wave activity calculation failed")
        raise

    logger.success(
        "Completed: points={}, activity rows={}, skipped weather rows={}, db={}",
        stats.point_count,
        stats.activity_count,
        stats.skipped_weather_rows,
        output,
    )


if __name__ == "__main__":
    main()
