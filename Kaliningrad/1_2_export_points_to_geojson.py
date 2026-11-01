from __future__ import annotations

"""
pypeline/2_1_export_points_to_geojson.py

Экспортирует точки береговой линии из SQLite-базы в GeoJSON-файл.

Запуск:
    python pypeline/2_1_export_points_to_geojson.py

По умолчанию берёт последний (наибольший id) источник с именем
SOURCE_NAME. При необходимости можно уточнить STRATEGY_NAME и
SOURCE_MODE, либо использовать конкретный SOURCE_ID.
"""

import os
from pathlib import Path

from loguru import logger

# --- Конфигурация --------------------------------------------------------

# Путь к SQLite-базе (если не задан через env)
DATABASE_PATH = Path("data/db/coastline.db")

# Куда сохранять GeoJSON
OUTPUT_PATH = Path("data/coastline/exported_points.geojson")

# Имя источника (поле `name` в таблице coastline_sources)
SOURCE_NAME = "KaliningradOSM"

# Опциональные фильтры (None = не фильтровать)
STRATEGY_NAME: str | None = None   # напр. "EqualStepAlongLineStrategy"
SOURCE_MODE: str | None = None     # напр. "all_lines"

# Если хотите экспортировать конкретный источник по id — задайте здесь
# (при SOURCE_ID is not None параметры выше игнорируются)
SOURCE_ID: int | None = None

# -------------------------------------------------------------------------


def setup_env() -> None:
    """Выставляет COASTLINE_DATABASE_URL, если не задан в окружении."""
    if "COASTLINE_DATABASE_URL" not in os.environ:
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL set to: {url}")


def main() -> None:
    setup_env()

    # Импортируем после выставления env, чтобы db-модуль подхватил URL
    from src.coastline.exporters.SQLiteToGeoJsonExporter import (
        SQLiteToGeoJsonExporter,
    )

    exporter = SQLiteToGeoJsonExporter()

    # Покажем что есть в БД
    sources = exporter.list_sources()
    if not sources:
        logger.warning("No sources found in the database. Run import pipeline first.")
        return

    logger.info(f"Sources in DB ({len(sources)}):")
    for s in sources:
        logger.info(
            f"  id={s['id']}  name={s['name']!r}  "
            f"strategy={s['strategy_name']}  "
            f"mode={s['source_mode']}  "
            f"pts={s['points_count']}  "
            f"crs={s['crs']}  "
            f"created={s['created_at']}"
        )

    # Экспорт
    if SOURCE_ID is not None:
        logger.info(f"Exporting source id={SOURCE_ID} → {OUTPUT_PATH}")
        result = exporter.export_by_id(
            source_id=SOURCE_ID,
            output_path=OUTPUT_PATH,
        )
    else:
        logger.info(
            f"Exporting source name={SOURCE_NAME!r} "
            f"strategy={STRATEGY_NAME!r} mode={SOURCE_MODE!r} → {OUTPUT_PATH}"
        )
        result = exporter.export(
            source_name=SOURCE_NAME,
            output_path=OUTPUT_PATH,
            strategy_name=STRATEGY_NAME,
            source_mode=SOURCE_MODE,
        )

    logger.success(f"Done. GeoJSON saved to: {result.resolve()}")


if __name__ == "__main__":
    main()
