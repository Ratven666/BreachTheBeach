from __future__ import annotations

"""Шаг 6.2: отчёт по статистике береговых точек — карты, графики, HTML, GeoJSON, выводы."""

import sys
from pathlib import Path

from loguru import logger


def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent
    return next(p for p in (directory, *directory.parents) if (p / "src/waves/coast_stats").is_dir())


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from src.waves.coast_stats import build_report  # noqa: E402

# ==================== НАСТРОЙКИ ====================

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"
STATS_DB = DATA_DIR / "db/wave_coast_stats.db"
REPORT_DIR = DATA_DIR / "wave/coast_stats_report"

BACKGROUND_GEOJSON: Path | None = None   # подложка суши для статичных карт (необязательно)
MAX_HTML_POINTS = 50_000                 # больше — случайная выборка точек на интерактивной карте
DPI = 150

if __name__ == "__main__":
    files = build_report(STATS_DB, REPORT_DIR, background_geojson=BACKGROUND_GEOJSON,
                         max_html_points=MAX_HTML_POINTS, dpi=DPI)
    logger.info("Откройте в браузере: {}", files["html"])
