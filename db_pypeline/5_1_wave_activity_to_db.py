from __future__ import annotations

"""Расчёт дневной волновой активности и запись в SQLite.

Предварительно должны быть выполнены этапы 0-4 db_pypeline.

Переменные окружения:
    WAVE_WEATHER_DB          — путь к weather_compact.db
    WAVE_COASTLINE_DB        — путь к coastline.db
    WAVE_BATHY_MANIFEST      — путь к bathymetry/manifest.json
    WAVE_OUTPUT_DB           — путь к выходному wave_activity.db
    WAVE_NORMAL_SOURCE_ID    — id источника нормалей (опционально)
    WAVE_START_DATE          — фильтр с даты YYYY-MM-DD (опционально)
    WAVE_END_DATE            — фильтр по дату YYYY-MM-DD (опционально)
    WAVE_OVERWATER_FACTOR    — коэффициент разгона ветра над водой (default: 1.1)
    WAVE_BREAKING_COEFF      — коэффициент обрушения волн γ_b (default: 0.55)
    WAVE_RHO_WATER           — плотность воды, кг/м³ (default: 1025.0)
    WAVE_GRAVITY             — ускорение свободного падения (default: 9.81)
    WAVE_BATHY_STEPS         — число точек профиля батиметрии (default: 200)
    WAVE_COMMIT_BATCH_SIZE   — размер батча записи в БД (default: 5000)
    WAVE_OVERWRITE           — перезаписать существующую БД (default: false)
    WAVE_LOG_LEVEL           — уровень логирования (default: INFO)
"""

import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger

# ── Корень проекта ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "db_pypeline" / "data"

# ── Конфигурация из переменных окружения ──────────────────────────────────────
WEATHER_DB = Path(
    os.getenv("WAVE_WEATHER_DB", DATA_DIR / "db" / "weather_compact.db")
)
COASTLINE_DB = Path(
    os.getenv("WAVE_COASTLINE_DB", DATA_DIR / "db" / "coastline.db")
)
BATHY_MANIFEST = Path(
    os.getenv("WAVE_BATHY_MANIFEST", DATA_DIR / "bathymetry" / "manifest.json")
)
OUTPUT_DB = Path(
    os.getenv("WAVE_OUTPUT_DB", DATA_DIR / "db" / "wave_activity.db")
)

NORMAL_SOURCE_ID = (
    int(os.environ["WAVE_NORMAL_SOURCE_ID"])
    if os.getenv("WAVE_NORMAL_SOURCE_ID")
    else None
)

START_DATE = os.getenv("WAVE_START_DATE") or None
END_DATE   = os.getenv("WAVE_END_DATE")   or None

OVERWATER_FACTOR  = float(os.getenv("WAVE_OVERWATER_FACTOR",  "1.1"))
BREAKING_COEFF    = float(os.getenv("WAVE_BREAKING_COEFF",    "0.55"))
RHO_WATER         = float(os.getenv("WAVE_RHO_WATER",         "1025.0"))
GRAVITY           = float(os.getenv("WAVE_GRAVITY",           "9.81"))
BATHY_STEPS       = int(os.getenv("WAVE_BATHY_STEPS",         "200"))
COMMIT_BATCH_SIZE = int(os.getenv("WAVE_COMMIT_BATCH_SIZE",   "5000"))

OVERWRITE  = os.getenv("WAVE_OVERWRITE", "false").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL  = os.getenv("WAVE_LOG_LEVEL", "INFO")


# ── Утилиты ───────────────────────────────────────────────────────────────────

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
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()

    # Проверяем наличие входных файлов ДО дорогостоящих импортов
    for path, label in (
        (WEATHER_DB,    "weather archive"),
        (COASTLINE_DB,  "coastline database"),
        (BATHY_MANIFEST,"bathymetry manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label}: {path.expanduser().resolve()}"
            )

    output    = OUTPUT_DB.expanduser().resolve()
    temporary = output.with_name(f"{output.name}.tmp")

    if output.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output database already exists: {output}. "
            "Set WAVE_OVERWRITE=true to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _remove_sqlite_files(temporary)

    # Передаём путь к coastline.db через переменную окружения до импорта ORM
    os.environ["COASTLINE_DATABASE_URL"] = (
        f"sqlite:///{COASTLINE_DB.expanduser().resolve().as_posix()}"
    )

    # Ленивые импорты — ORM инициализируется только после выставления URL
    from src.bathymetry.archive import BathymetryArchive
    from src.coastline.storage.db import SessionLocal
    from src.weather_history.archive import CompactWeatherRepository
    from src.waves.services.wave_activity_db_service import (
        WaveActivityDatabaseBuilder,
        resolve_normal_source_id,
    )
    from src.waves.storage import WaveActivityRepository

    logger.info("Weather archive: {}",    WEATHER_DB.resolve())
    logger.info("Coastline database: {}", COASTLINE_DB.resolve())
    logger.info("Bathymetry manifest: {}", BATHY_MANIFEST.resolve())
    logger.info("Output database: {}",    output)
    logger.info(
        "Parameters: overwater_factor={}, breaking_coeff={}, "
        "rho_water={}, g={}, bathy_steps={}",
        OVERWATER_FACTOR, BREAKING_COEFF, RHO_WATER, GRAVITY, BATHY_STEPS,
    )

    try:
        with SessionLocal() as coastline_session:
            normal_source_id = resolve_normal_source_id(
                coastline_session,
                NORMAL_SOURCE_ID,
            )
            logger.info("Selected normal_source_id={}", normal_source_id)

            bathymetry = BathymetryArchive(BATHY_MANIFEST)

            with CompactWeatherRepository(WEATHER_DB) as weather_repo:
                with WaveActivityRepository(temporary) as output_repo:
                    stats = WaveActivityDatabaseBuilder(
                        coastline_session=coastline_session,
                        weather_repository=weather_repo,
                        bathymetry_archive=bathymetry,
                        output_repository=output_repo,
                        normal_source_id=normal_source_id,
                        start_date=_validate_date(START_DATE),
                        end_date=_validate_date(END_DATE),
                        overwater_factor=OVERWATER_FACTOR,
                        breaking_coeff=BREAKING_COEFF,
                        rho_water=RHO_WATER,
                        g=GRAVITY,
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
        "Completed: points={}, processed_days={}, active_rows={}, "
        "skipped_weather={}, land_sector_days={}, missing_bathymetry_days={}, "
        "db={}",
        stats.point_count,
        stats.processed_day_count,
        stats.activity_count,
        stats.skipped_weather_rows,
        stats.land_sector_rows,
        stats.missing_bathymetry_rows,
        output,
    )
    logger.info("Next step: python db_pypeline/5_2_wave_exposure_to_db.py")


if __name__ == "__main__":
    main()