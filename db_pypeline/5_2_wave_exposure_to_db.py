from __future__ import annotations

"""Расчёт WEI-показателей, рангов R1-R4 и индекса WER.

Использует wave_activity.db, созданную скриптом:

    db_pypeline/5_1_wave_activity_to_db.py

Для каждой точки рассчитываются:

    mean_cwef_wm
        Среднее дневное CWEF, Вт/м.

    e_storm_mjm
        Энергия дней выше индивидуального штормового порога,
        МДж/м.

    k_dir
        Доля CWEF, приходящаяся на наиболее энергетически
        значимое окно из трёх соседних 16-румбовых секторов.

    cv
        Коэффициент вариации дневного CWEF.

Затем по ансамблю всех точек рассчитываются квинтильные ранги:

    R1 = rank(mean_cwef_wm)
    R2 = rank(e_storm_mjm)
    R3 = rank(k_dir)
    R4 = rank(cv)

Итоговый индекс:

    WER = (R1 * R2 * R3 * R4) ** 0.25

Результат записывается в таблицу wave_exposure_index.

Переменные окружения:
    WAVE_ACTIVITY_DB
    WAVE_INDEX_OUTPUT_DB
    WAVE_STORM_PERCENTILE
    WAVE_STORM_THRESHOLD
    WAVE_DT_SECONDS
    WAVE_LOG_LEVEL
"""

import os
import shutil
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

OUTPUT_DB = Path(
    os.getenv(
        "WAVE_INDEX_OUTPUT_DB",
        ACTIVITY_DB,
    )
)

STORM_PERCENTILE = float(
    os.getenv(
        "WAVE_STORM_PERCENTILE",
        "90.0",
    )
)

_raw_threshold = os.getenv(
    "WAVE_STORM_THRESHOLD"
)

STORM_THRESHOLD = (
    None
    if (
        _raw_threshold is None
        or not _raw_threshold.strip()
    )
    else float(_raw_threshold)
)

DT_SECONDS = float(
    os.getenv(
        "WAVE_DT_SECONDS",
        "86400.0",
    )
)

LOG_LEVEL = os.getenv(
    "WAVE_LOG_LEVEL",
    "INFO",
)


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


def _remove_sqlite_files(
    path: Path,
) -> None:
    for candidate in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    configure_logging()

    source = (
        ACTIVITY_DB
        .expanduser()
        .resolve()
    )
    output = (
        OUTPUT_DB
        .expanduser()
        .resolve()
    )

    if not source.is_file():
        raise FileNotFoundError(
            "Wave activity database not found: "
            f"{source}. Run "
            "db_pypeline/"
            "5_1_wave_activity_to_db.py first."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_name(
        f"{output.name}.indices.tmp"
    )

    _remove_sqlite_files(
        temporary
    )

    logger.info(
        "Wave activity database: {}",
        source,
    )
    logger.info(
        "Output database: {}",
        output,
    )
    logger.info(
        "Storm percentile: {}",
        STORM_PERCENTILE,
    )
    logger.info(
        "Explicit storm threshold: {}",
        STORM_THRESHOLD,
    )
    logger.info(
        "Time step: {} seconds",
        DT_SECONDS,
    )

    try:
        # Работаем с копией, чтобы при любой ошибке
        # исходная БД осталась неизменной.
        shutil.copy2(
            source,
            temporary,
        )

        from src.waves.services.wave_exposure_db_service import (
            WaveExposureDatabaseBuilder,
        )
        from src.waves.storage import (
            WaveActivityRepository,
        )

        with WaveActivityRepository(
            temporary
        ) as repository:
            stats = (
                WaveExposureDatabaseBuilder(
                    repository=repository,
                    storm_percentile=(
                        STORM_PERCENTILE
                    ),
                    storm_threshold=(
                        STORM_THRESHOLD
                    ),
                    dt_seconds=DT_SECONDS,
                ).run()
            )

        if output.exists():
            _remove_sqlite_files(
                output
            )

        os.replace(
            temporary,
            output,
        )

    except Exception:
        _remove_sqlite_files(
            temporary
        )
        logger.exception(
            "Wave exposure index "
            "calculation failed"
        )
        raise

    logger.success(
        "Completed: points={}, "
        "valid_WER={}, "
        "min_WER={}, "
        "max_WER={}, "
        "mean_WER={}, "
        "db={}",
        stats.point_count,
        stats.valid_index_count,
        stats.min_wer,
        stats.max_wer,
        stats.mean_wer,
        output,
    )

    logger.info(
        "Table wave_exposure_index contains:\n"
        "  mean_cwef_wm\n"
        "  e_storm_mjm\n"
        "  storm_threshold_wm\n"
        "  storm_percentile\n"
        "  k_dir\n"
        "  cv\n"
        "  n_days\n"
        "  n_storm_days\n"
        "  top3_sectors\n"
        "  r1, r2, r3, r4\n"
        "  wer"
    )


if __name__ == "__main__":
    main()
