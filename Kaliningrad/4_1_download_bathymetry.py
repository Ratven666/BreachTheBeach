from __future__ import annotations

"""
Построение локального архива батиметрии по точкам и fetch из БД.

Предварительная последовательность:

    python db_pypeline/1_1_import_coastline_points.py
    python db_pypeline/2_1_normals_to_db.py
    python db_pypeline/3_1_wind_fetch_to_db.py
    python db_pypeline/4_1_download_bathymetry.py

Переменные окружения переопределяют константы ниже:

    COASTLINE_DATABASE_URL           — SQLAlchemy URL базы данных
    BATHY_POINT_SOURCE_ID            — id набора точек (int)
    BATHY_OUTPUT_DIR                 — каталог результата
    BATHY_PROVIDER                   — auto | emodnet | gebco
    BATHY_SAFETY_FACTOR              — множитель к максимальному fetch
    BATHY_PADDING_M                  — дополнительный буфер bbox в метрах
    BATHY_MAX_CELLS_PER_TILE         — максимальное число ячеек тайла
    BATHY_PLANNING_RESOLUTION_ARCSEC — разрешение для оценки тайлов
    BATHY_OVERLAP_POINTS             — перекрытие соседних тайлов в точках
    BATHY_RETRIES                    — число попыток загрузки
    BATHY_FORCE                      — 1/true/yes/on: перезаписать тайлы
    BATHY_MAX_FETCH_M                — ограничение максимального fetch (м);
                                       если не задано — берётся из БД как есть
"""

import math
import os
import sys
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Пути и настройки по умолчанию — редактируйте здесь
# ---------------------------------------------------------------------------

# Корень проекта — три уровня вверх от db_pypeline/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Тот же путь, что использует db.py по умолчанию
DATABASE_PATH = (
    _PROJECT_ROOT / "Kaliningrad" / "data" / "db" / "coastline.db"
)


def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        database_url = f"sqlite:///{DATABASE_PATH.as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = database_url
        logger.debug(f"COASTLINE_DATABASE_URL -> {database_url}")

# Каталог для сохранения тайлов и manifest.json.
DEFAULT_OUTPUT_DIR: Path = _PROJECT_ROOT / "Kaliningrad" / "data" / "bathymetry"

# id источника береговых точек; None → последний с рассчитанными fetch.
DEFAULT_POINT_SOURCE_ID: int | None = None

# Источник батиметрии: "auto" | "emodnet" | "gebco".
DEFAULT_PROVIDER: str = "auto"

# Коэффициент запаса относительно максимального fetch_length_m.
DEFAULT_SAFETY_FACTOR: float = 1.05

# Дополнительный геодезический буфер вокруг bbox тайла, метры.
DEFAULT_PADDING_M: float = 1_000.0

# Порог числа ячеек одного запроса; при превышении — новый тайл.
DEFAULT_MAX_CELLS_PER_TILE: int = 10_000_000

# Угловые секунды: используется только для оценки, а не для скачивания.
# EMODnet: 3.75″ (1/16 угловой минуты); GEBCO: 15″.
DEFAULT_PLANNING_RESOLUTION_ARCSEC: float = 3.75

# Число береговых точек, общих для двух соседних тайлов.
DEFAULT_OVERLAP_POINTS: int = 3

# Число повторных попыток при сбое загрузки провайдера.
DEFAULT_RETRIES: int = 3

# True → перезаписывать уже скачанные .nc-файлы.
DEFAULT_FORCE: bool = False

# Максимальный fetch в метрах: None → брать из БД без ограничений.
# Задайте значение, чтобы ограничить радиус покрытия сверху
# (например, 50_000.0 для мелководных закрытых акваторий).
DEFAULT_MAX_FETCH_M: float | None = None

# ---------------------------------------------------------------------------


sys.path.insert(0, str(_PROJECT_ROOT))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env_optional(
    name: str,
    default: float | None,
) -> float | None:
    """
    Возвращает float из переменной окружения или default.

    Пустая строка и "none" / "null" трактуются как None.
    """
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip().lower()
    if stripped in {"", "none", "null"}:
        return None
    parsed = float(stripped)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(
            f"{name}={value!r} — должно быть конечным положительным числом"
        )
    return parsed


def _setup_database_url() -> None:
    """
    Устанавливает COASTLINE_DATABASE_URL из DEFAULT_DATABASE_PATH,
    если переменная окружения не задана явно.
    """
    if "COASTLINE_DATABASE_URL" in os.environ:
        return
    url = f"sqlite:///{DATABASE_PATH.as_posix()}"
    os.environ["COASTLINE_DATABASE_URL"] = url
    logger.debug(f"COASTLINE_DATABASE_URL -> {url}")


def main() -> None:
    _setup_database_url()

    from src.bathymetry.archive import (
        BathymetryArchiveBuilder,
        BathymetryArchiveConfig,
    )
    from src.coastline.storage import db as database
    from src.coastline.storage.models import Base

    # Переменные окружения имеют приоритет над константами выше.
    source_env = os.getenv("BATHY_POINT_SOURCE_ID")
    point_source_id: int | None = (
        int(source_env) if source_env else DEFAULT_POINT_SOURCE_ID
    )

    provider = os.getenv("BATHY_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider not in {"auto", "emodnet", "gebco"}:
        raise ValueError(
            f"BATHY_PROVIDER={provider!r} — допустимые значения: auto, emodnet, gebco"
        )

    max_fetch_override = _float_env_optional(
        "BATHY_MAX_FETCH_M", DEFAULT_MAX_FETCH_M
    )

    config = BathymetryArchiveConfig(
        output_dir=Path(
            os.getenv("BATHY_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
        ),
        provider=provider,
        safety_factor=float(
            os.getenv("BATHY_SAFETY_FACTOR", str(DEFAULT_SAFETY_FACTOR))
        ),
        padding_m=float(
            os.getenv("BATHY_PADDING_M", str(DEFAULT_PADDING_M))
        ),
        max_cells_per_tile=int(
            os.getenv("BATHY_MAX_CELLS_PER_TILE", str(DEFAULT_MAX_CELLS_PER_TILE))
        ),
        planning_resolution_arcsec=float(
            os.getenv(
                "BATHY_PLANNING_RESOLUTION_ARCSEC",
                str(DEFAULT_PLANNING_RESOLUTION_ARCSEC),
            )
        ),
        overlap_points=int(
            os.getenv("BATHY_OVERLAP_POINTS", str(DEFAULT_OVERLAP_POINTS))
        ),
        retries=int(
            os.getenv("BATHY_RETRIES", str(DEFAULT_RETRIES))
        ),
        force=_bool_env("BATHY_FORCE", DEFAULT_FORCE),
        max_fetch_override_m=max_fetch_override,
    )

    if config.max_fetch_override_m is not None:
        logger.info(
            f"max_fetch_override_m={config.max_fetch_override_m:.0f} м — "
            "fetch из БД будет ограничен этим значением"
        )

    Base.metadata.create_all(database.engine)

    with database.SessionLocal() as session:
        manifest_path = BathymetryArchiveBuilder(
            session=session,
            config=config,
        ).build(point_source_id)

    logger.success(f"Bathymetry download completed: {manifest_path}")


if __name__ == "__main__":
    main()
