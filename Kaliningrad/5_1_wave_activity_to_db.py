from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"

WEATHER_DB = Path(os.getenv("WAVE_WEATHER_DB", DATA_DIR / "db" / "weather_compact.db"))
COASTLINE_DB = Path(os.getenv("WAVE_COASTLINE_DB", DATA_DIR / "db" / "coastline.db"))
BATHY_MANIFEST = Path(os.getenv("WAVE_BATHY_MANIFEST", DATA_DIR / "bathymetry" / "manifest.json"))
OUTPUT_DB = Path(os.getenv("WAVE_OUTPUT_DB", DATA_DIR / "db" / "wave_activity.db"))

NORMAL_SOURCE_ID = (
    int(os.environ["WAVE_NORMAL_SOURCE_ID"])
    if os.getenv("WAVE_NORMAL_SOURCE_ID")
    else None
)
START_DATE = os.getenv("WAVE_START_DATE") or None
END_DATE = os.getenv("WAVE_END_DATE") or None
OVERWATER_FACTOR = float(os.getenv("WAVE_OVERWATER_FACTOR", "1.1"))
BREAKING_COEFF = float(os.getenv("WAVE_BREAKING_COEFF", "0.55"))
RHO_WATER = float(os.getenv("WAVE_RHO_WATER", "1025.0"))
GRAVITY = float(os.getenv("WAVE_GRAVITY", "9.81"))
BATHY_STEPS = int(os.getenv("WAVE_BATHY_STEPS", "1000"))
COMMIT_BATCH_SIZE = int(os.getenv("WAVE_COMMIT_BATCH_SIZE", "500"))
# WAVE_MAX_WORKERS=0 → автоматически (cpu_count - 1).
MAX_WORKERS_RAW = int(os.getenv("WAVE_MAX_WORKERS", "10"))
OVERWRITE = os.getenv("WAVE_OVERWRITE", "false").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("WAVE_LOG_LEVEL", "INFO")


def _parse_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be in ISO format YYYY-MM-DD, got {value!r}") from exc


def _resolve_max_workers(value: int) -> int | None:
    if value < 0:
        raise ValueError(f"WAVE_MAX_WORKERS must be >= 0, got {value}")
    return value or None


def _remove_sqlite_files(path: Path) -> None:
    for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal")):
        if p.exists():
            p.unlink()


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

    start_date = _parse_date(START_DATE, "WAVE_START_DATE")
    end_date = _parse_date(END_DATE, "WAVE_END_DATE")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError(
            f"WAVE_START_DATE ({start_date}) is later than WAVE_END_DATE ({end_date})"
        )

    max_workers = _resolve_max_workers(MAX_WORKERS_RAW)

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
            f"Output DB already exists: {output}. Set WAVE_OVERWRITE=true to replace."
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

    logger.info("Weather archive:     {}", WEATHER_DB.resolve())
    logger.info("Coastline database:  {}", COASTLINE_DB.resolve())
    logger.info("Bathymetry manifest: {}", BATHY_MANIFEST.resolve())
    logger.info("Output database:     {}", output)
    logger.info(
        "Parameters: period={}..{}, overwater_factor={}, breaking_coeff={}, "
        "rho_water={}, g={}, bathy_steps={}, max_workers={}",
        start_date or "begin", end_date or "end",
        OVERWATER_FACTOR, BREAKING_COEFF, RHO_WATER, GRAVITY, BATHY_STEPS,
        max_workers or "auto",
    )

    try:
        with SessionLocal() as coastline_session:
            normal_source_id = resolve_normal_source_id(coastline_session, NORMAL_SOURCE_ID)
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
                        start_date=start_date,
                        end_date=end_date,
                        overwater_factor=OVERWATER_FACTOR,
                        breaking_coeff=BREAKING_COEFF,
                        rho_water=RHO_WATER,
                        g=GRAVITY,
                        bathy_n_steps=BATHY_STEPS,
                        commit_batch_size=COMMIT_BATCH_SIZE,
                        max_workers=max_workers,
                    ).run()

        if output.exists():
            _remove_sqlite_files(output)
        os.replace(temporary, output)

    except Exception:
        _remove_sqlite_files(temporary)
        logger.exception("Wave activity calculation failed")
        raise

    if stats.error_point_count > 0:
        logger.warning(
            "{} of {} points failed during calculation and have empty summaries",
            stats.error_point_count, stats.point_count,
        )

    logger.success(
        "Completed: points={}, processed_days={}, active_rows={}, "
        "skipped_weather={}, land_sector_days={}, missing_bathymetry_days={}, "
        "calc_errors={}, db={}",
        stats.point_count, stats.processed_day_count, stats.activity_count,
        stats.skipped_weather_rows, stats.land_sector_rows,
        stats.missing_bathymetry_rows, stats.error_point_count, output,
    )
    logger.info("Next: python db_pypeline/5_2_wave_exposure_to_db.py")


if __name__ == "__main__":
    main()
