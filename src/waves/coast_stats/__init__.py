"""Статистика волновой активности по береговым точкам: тренды, сезонность, участки, отчёт."""

from .config import CLASS_LABELS, METRICS, StatsConfig
from .report import build_report
from .service import build_coast_stats

__all__ = ["CLASS_LABELS", "METRICS", "StatsConfig", "build_coast_stats", "build_report"]
