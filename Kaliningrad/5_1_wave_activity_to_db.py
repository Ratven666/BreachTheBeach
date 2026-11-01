from __future__ import annotations

import fcntl
import json
import math
import os
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"


def env_path(
    name: str,
    default: Path,
) -> Path:
    return (
        Path(os.getenv(name, str(default)))
        .expanduser()
        .resolve()
    )


def env_date(
    name: str,
) -> date | None:
    value = os.getenv(name)

    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{name}: expected YYYY-MM-DD, got {value!r}"
        ) from exc


def positive_int(
    name: str,
    default: int,
    minimum: int = 1,
) -> int:
    value = int(os.getenv(name, str(default)))

    if value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    return value


def positive_float(
    name: str,
    default: float,
) -> float:
    value = float(os.getenv(name, str(default)))

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be finite and > 0"
        )

    return value


@contextmanager
def output_lock(
    path: Path,
):
    # macOS/Linux.
    #
    # Файл блокировки намеренно не удаляется:
    # блокировку снимает ОС при завершении процесса.
    with Path(f"{path}.lock").open("a+") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another wave calculation owns {path}"
            ) from exc

        try:
            yield
        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


def main() -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        level=os.getenv("WAVE_LOG_LEVEL", "INFO"),
    )

    weather = env_path(
        "WAVE_WEATHER_DB",
        DATA_DIR / "db/weather_compact.db",
    )

    coastline = env_path(
        "WAVE_COASTLINE_DB",
        DATA_DIR / "db/coastline.db",
    )

    manifest = env_path(
        "WAVE_BATHY_MANIFEST",
        DATA_DIR / "bathymetry/manifest.json",
    )

    output = env_path(
        "WAVE_OUTPUT_DB",
        DATA_DIR / "db/wave_activity.db",
    )

    start = env_date("WAVE_START_DATE")
    end = env_date("WAVE_END_DATE")

    if (
        start is not None
        and end is not None
        and start > end
    ):
        raise ValueError(
            "WAVE_START_DATE is later than WAVE_END_DATE"
        )

    max_workers = positive_int(
        "WAVE_MAX_WORKERS",
        4,
        minimum=0,
    )

    max_workers = max_workers or max(
        1,
        (os.cpu_count() or 1) - 1,
    )

    pool_points = positive_int(
        "WAVE_POOL_BATCH_POINTS",
        128,
    )

    commit_rows = positive_int(
        "WAVE_COMMIT_BATCH_SIZE",
        500,
    )

    commit_points = positive_int(
        "WAVE_COMMIT_POINTS",
        4,
    )

    commit_seconds = positive_float(
        "WAVE_COMMIT_SECONDS",
        10.0,
    )

    breaking = positive_float(
        "WAVE_BREAKING_COEFF",
        0.55,
    )

    if breaking > 1:
        raise ValueError(
            "WAVE_BREAKING_COEFF must be <= 1"
        )

    options = {
        "start_date": (
            start.isoformat()
            if start
            else None
        ),
        "end_date": (
            end.isoformat()
            if end
            else None
        ),
        "bathy_manifest_path": str(manifest),
        "bathy_n_steps": positive_int(
            "WAVE_BATHY_STEPS",
            1000,
            minimum=2,
        ),
        "overwater_factor": positive_float(
            "WAVE_OVERWATER_FACTOR",
            1.1,
        ),
        "breaking_coeff": breaking,
        "rho_water": positive_float(
            "WAVE_RHO_WATER",
            1025.0,
        ),
        "g": positive_float(
            "WAVE_GRAVITY",
            9.81,
        ),
    }

    overwrite = (
        os.getenv("WAVE_OVERWRITE", "false").lower()
        in {"1", "true", "yes", "on"}
    )

    raw_source = os.getenv(
        "WAVE_NORMAL_SOURCE_ID"
    )

    source_id = (
        int(raw_source)
        if raw_source
        else None
    )

    for path in (
        weather,
        coastline,
        manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    content = json.loads(
        manifest.read_text(encoding="utf-8")
    )

    tiles = [
        (manifest.parent / tile["file"]).resolve()
        for tile in content["tiles"]
    ]

    inputs = [
        weather,
        coastline,
        manifest,
        *tiles,
    ]

    if output in inputs:
        raise ValueError(
            "WAVE_OUTPUT_DB must not overwrite an input file"
        )

    # Входные архивы должны быть неизменяемыми во время расчёта.
    # Нельзя игнорировать данные, оставшиеся в их WAL.
    for db in (
        weather,
        coastline,
    ):
        wal = Path(f"{db}-wal")

        if wal.exists() and wal.stat().st_size:
            raise RuntimeError(
                "Checkpoint/close the input SQLite WAL first: "
                f"{wal}"
            )

    os.environ["COASTLINE_DATABASE_URL"] = (
        f"sqlite:///{coastline.as_posix()}"
    )

    # Устанавливается до импорта NumPy/SciPy.
    # Не даём каждому worker создавать собственный
    # многопоточный BLAS-пул.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    from sqlalchemy import select

    from src.bathymetry.archive import BathymetryArchive
    from src.coastline.storage.db import SessionLocal
    from src.coastline.storage.models import (
        CoastlineNormalSourceModel,
    )
    from src.waves.services.wave_activity_db_service import (
        load_wave_points,
        resolve_normal_source_id,
    )
    from src.waves.services.resumable_wave_activity import (
        ALGORITHM_VERSION,
        CheckpointRepository,
        input_signature,
        run_incremental,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_lock(output):
        # SHA-256 входных архивов и исходного кода.
        #
        # Параметры производительности — число workers,
        # размеры пачек — в fingerprint не входят.
        # Их можно менять при продолжении расчёта.
        code = sorted(
            (PROJECT_ROOT / "src").rglob("*.py")
        )

        code += [
            Path(__file__).resolve(),
            PROJECT_ROOT / "poetry.lock",
        ]

        signature, states = input_signature(
            inputs + code
        )

        with SessionLocal() as session:
            source_id = resolve_normal_source_id(
                session,
                source_id,
            )

            points = load_wave_points(
                session,
                source_id,
            )

            if not points:
                raise ValueError(
                    "No coastline points found"
                )

            point_source = session.execute(
                select(
                    CoastlineNormalSourceModel.point_source_id
                )
                .where(
                    CoastlineNormalSourceModel.id
                    == source_id
                )
            ).scalar_one()

            archive = BathymetryArchive(
                manifest
            )

            if archive.point_source_id != point_source:
                raise ValueError(
                    "Bathymetry and normals refer "
                    "to different point sources"
                )

        config = {
            "algorithm_version": ALGORITHM_VERSION,
            "normal_source_id": source_id,
            "point_count": len(points),
            "options": options,
            "inputs": signature,
        }

        with CheckpointRepository(output) as repository:
            pending = repository.setup(
                config,
                points,
                overwrite=overwrite,
            )

            logger.info(
                "points={}, saved={}, pending={}, "
                "workers={}, database={}",
                len(points),
                len(points) - len(pending),
                len(pending),
                max_workers,
                output,
            )

            stats = run_incremental(
                repository,
                pending,
                weather_path=weather,
                coastline_path=coastline,
                options=options,
                input_states=states,
                max_workers=max_workers,
                pool_batch_points=pool_points,
                commit_batch_size=commit_rows,
                commit_points=commit_points,
                commit_seconds=commit_seconds,
            )

        logger.info(
            "Saved stats: {}",
            stats,
        )

        if stats.error_point_count:
            raise SystemExit(
                "Some points failed. Fix the cause "
                "and restart WITHOUT WAVE_OVERWRITE."
            )

        logger.success(
            "Completed: {}",
            output,
        )


if __name__ == "__main__":
    main()
