from __future__ import annotations

"""Параметры расчёта статистики волновой активности по береговым точкам."""

from dataclasses import dataclass
from pathlib import Path

# Сезоны: индексы месяцев (0 = январь). Декабрь относится к зиме следующего года.
SEASONS: dict[str, tuple[int, ...]] = {
    "djf": (11, 0, 1), "mam": (2, 3, 4), "jja": (5, 6, 7), "son": (8, 9, 10),
}

# Ряды, для которых оценивается тренд: ключ -> подпись.
METRICS: dict[str, str] = {
    "mean": "Средняя мощность за год",
    "storm_days": "Число штормовых дней",
    "max": "Годовой максимум",
    "active_days": "Число активных дней",
    "djf": "Зима (XII–II)",
    "mam": "Весна (III–V)",
    "jja": "Лето (VI–VIII)",
    "son": "Осень (IX–XI)",
}
SECTION_METRICS: tuple[str, ...] = ("mean", "storm_days", "djf", "mam", "jja", "son")

CLASS_LABELS: dict[int, str] = {
    2: "сильный рост", 1: "рост", 0: "без изменений", -1: "снижение", -2: "сильное снижение",
}
ALL_SECTION = "Все точки"  # сводная строка в таблицах участков (section_id = -1)


@dataclass(frozen=True)
class StatsConfig:
    activity_db: Path | str                  # wave_activity.db (шаг 5.1) или SQLAlchemy URL
    output_db: Path | str                    # БД результатов (SQLite-файл или SQLAlchemy URL)

    points_geojson: Path | None = None       # None — все точки; иначе только точки из файла
    match_tolerance_m: float = 50.0          # привязка по координатам, если в файле нет point_id / id

    sections_geojson: Path | None = None     # полигоны участков; None — регулярная сетка
    section_name_field: str = "name"
    section_grid_deg: tuple[float, float] = (10.0, 5.0)   # шаг сетки: долгота, широта

    year_from: int | None = None             # ограничение периода (используются полные годы)
    year_to: int | None = None

    storm_percentile: float = 90.0           # порог шторма: перцентиль по всем дням точки
    alpha: float = 0.05                      # уровень значимости (после поправки FDR)
    min_pct_per_decade: float = 1.0          # практически значимый тренд, % за 10 лет
    strong_pct_per_decade: float = 5.0       # «сильное» изменение, % за 10 лет
    prewhiten: bool = True                   # TFPW: устранение автокорреляции перед тестом
    epochs: tuple[tuple[int, int], tuple[int, int]] | None = None  # None — две половины ряда
    return_periods: tuple[int, ...] = (10, 50)

    chunk_points: int = 200                  # точек в одном запросе к БД
    n_workers: int = 4                       # процессов; 1 — без пула
    save_point_annual: bool = False          # сохранять годовые ряды каждой точки
    