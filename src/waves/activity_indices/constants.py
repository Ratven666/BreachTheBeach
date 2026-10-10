from __future__ import annotations

"""Константы модуля индексов волновой активности.

Дни во всех БД проекта — целые смещения от EPOCH (1940-01-01 = день 0).
"""

from src.weather_history.archive.schema import EPOCH, MAX_DAY_OFFSET

SCHEMA_VERSION: int = 1
SOURCE_SCHEMA_VERSION: int = 3

SECONDS_PER_DAY: float = 86_400.0
JOULES_PER_MJ: float = 1.0e6
DAYS_PER_YEAR: float = 365.2425

MAX_ANGLE_DEG: float = 85.0
SIGN_EPSILON_WM: float = 1.0e-9
ROUND_DIGITS: int = 6

MIN_PERIODS_FOR_TREND: int = 3
MIN_PERIODS_FOR_CV: int = 2

TREND_ONLY_COMPLETE_PERIODS: bool = True

SQLITE_CACHE_KIB: int = 65_536

GEOJSON_MODE_TRENDS: str = "trends"
GEOJSON_MODE_PERIODS: str = "periods"
GEOJSON_MODES: tuple[str, ...] = (GEOJSON_MODE_TRENDS, GEOJSON_MODE_PERIODS)

ALONGSHORE_SIGN_CONVENTION: str = (
    "theta = wrap180(wind_azimuth - normal_azimuth); "
    "positive: wave source is clockwise from the shore normal"
)

__all__ = [name for name in dir() if name.isupper()] + ["EPOCH", "MAX_DAY_OFFSET"]
