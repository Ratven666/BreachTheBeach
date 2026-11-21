from __future__ import annotations

"""Чтение wave_activity.db: точки, календарь, суточные ряды, разбиение на участки."""

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree
from sqlalchemy import Column, Engine, Float, Integer, MetaData, Table, create_engine, func, select
from sqlalchemy.engine import URL

from src.weather_history.archive.schema import EPOCH

from .config import StatsConfig

metadata = MetaData()
wave_points = Table(
    "wave_points", metadata,
    Column("id", Integer, primary_key=True), Column("lon", Float), Column("lat", Float),
    Column("normal_azimuth_deg", Integer),
)
wave_activity = Table(
    "wave_activity", metadata,
    Column("point_id", Integer, primary_key=True), Column("day", Integer, primary_key=True),
    Column("wind_azimuth_deg", Integer), Column("cwef_wm", Integer),
)
ROW_DTYPE = np.dtype([("p", "i8"), ("d", "i4"), ("w", "i2"), ("c", "f8")])
EARTH_RADIUS_M = 6_371_000.0


def make_engine(target: Path | str, *, readonly: bool = False) -> Engine:
    """Путь -> SQLite (для входной БД только чтение); строка с '://' -> готовый SQLAlchemy URL."""
    if "://" in str(target):
        return create_engine(str(target))
    path = Path(target).expanduser().resolve()
    if readonly:
        return create_engine(
            "sqlite://", creator=lambda: sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(URL.create("sqlite", database=str(path)))


# ==================== КАЛЕНДАРЬ ====================

@dataclass(frozen=True)
class Calendar:
    """Полные календарные годы анализа. Дни — смещения от EPOCH (1940-01-01 = 0)."""
    start: int                 # 1 января первого года
    end: int                   # 31 декабря последнего года
    years: np.ndarray          # [Y]
    ym: np.ndarray             # [n_days] -> индекс_года * 12 + индекс_месяца
    days: np.ndarray           # [Y, 12] число дней в каждом месяце

    @property
    def n_days(self) -> int:
        return self.end - self.start + 1


def load_calendar(engine: Engine, year_from: int | None, year_to: int | None) -> Calendar:
    with engine.connect() as con:
        first = con.execute(select(func.min(wave_activity.c.day))).scalar_one()
        last = con.execute(select(func.max(wave_activity.c.day))).scalar_one()
    d0, d1 = EPOCH + timedelta(days=first), EPOCH + timedelta(days=last)

    y0 = d0.year + (d0 != date(d0.year, 1, 1))          # только полные годы
    y1 = d1.year - (d1 != date(d1.year, 12, 31))
    y0, y1 = max(y0, year_from or y0), min(y1, year_to or y1)
    dates = pd.date_range(date(y0, 1, 1), date(y1, 12, 31), freq="D")
    ym = ((dates.year - y0) * 12 + dates.month - 1).to_numpy()
    return Calendar(
        start=(date(y0, 1, 1) - EPOCH).days, end=(date(y1, 12, 31) - EPOCH).days,
        years=np.arange(y0, y1 + 1), ym=ym, days=np.bincount(ym).reshape(-1, 12).astype(float),
    )


# ==================== ТОЧКИ ====================

def _unit_xyz(lon, lat) -> np.ndarray:
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def load_points(engine: Engine, points_geojson: Path | None, tolerance_m: float) -> pd.DataFrame:
    """Все точки БД либо только точки из GeoJSON (по point_id / id, иначе по координатам)."""
    with engine.connect() as con:
        points = pd.read_sql(select(wave_points.c.id.label("point_id"), wave_points.c.lon,
                                    wave_points.c.lat, wave_points.c.normal_azimuth_deg)
                             .order_by(wave_points.c.id), con)
    if points_geojson is None:
        return points

    selection = gpd.read_file(points_geojson).to_crs(4326)
    key = next((name for name in ("point_id", "id") if name in selection.columns), None)
    if key is not None:
        keep = points["point_id"].isin(selection[key].dropna().astype(int))
    else:
        geometry = selection.geometry.representative_point()
        distance, index = cKDTree(_unit_xyz(points["lon"], points["lat"])).query(
            _unit_xyz(geometry.x, geometry.y))
        index = index[distance * EARTH_RADIUS_M <= tolerance_m]
        keep = np.isin(np.arange(len(points)), index)
    logger.info("Выбрано точек по {}: {} из {}", Path(points_geojson).name, int(keep.sum()), len(selection))
    return points[keep].reset_index(drop=True)


def read_activity(engine: Engine, point_ids: np.ndarray, calendar: Calendar) -> np.ndarray:
    """Суточные записи группы точек в пределах календаря, порядок: point_id, day."""
    a = wave_activity.c
    query = (select(a.point_id, a.day, a.wind_azimuth_deg, a.cwef_wm)
             .where(a.point_id.in_([int(i) for i in point_ids]),
                    a.day.between(calendar.start, calendar.end))
             .order_by(a.point_id, a.day))
    with engine.connect() as con:                      # курсор DBAPI отдаёт кортежи — без обёрток Row
        return np.fromiter(con.execute(query).cursor, dtype=ROW_DTYPE)


# ==================== УЧАСТКИ ====================

def assign_sections(points: pd.DataFrame, cfg: StatsConfig) -> tuple[np.ndarray, list[str]]:
    """Номер участка для каждой точки и названия участков (в порядке номеров)."""
    if cfg.sections_geojson is not None:
        polygons = gpd.read_file(cfg.sections_geojson).to_crs(4326)[[cfg.section_name_field, "geometry"]]
        geo = gpd.GeoDataFrame(points[["point_id"]], crs=4326,
                               geometry=gpd.points_from_xy(points["lon"], points["lat"]))
        joined = gpd.sjoin(geo, polygons, how="left", predicate="within")
        name = joined.groupby(level=0)[cfg.section_name_field].first().fillna("Вне участков").astype(str)
        names = [n for n in [*polygons[cfg.section_name_field].astype(str).unique(), "Вне участков"]
                 if n in set(name)]
        return name.map({n: i for i, n in enumerate(names)}).to_numpy(), names

    # Регулярная сетка; долгота 0..360, чтобы Чукотка не разрывалась на 180°.
    step_lon, step_lat = cfg.section_grid_deg
    col = np.floor((points["lon"].to_numpy() % 360.0) / step_lon).astype(int)
    row = np.floor(points["lat"].to_numpy() / step_lat).astype(int)
    cells, index = np.unique(np.column_stack([col, row]), axis=0, return_inverse=True)

    def label(c: int, r: int) -> str:
        west, south = c * step_lon, r * step_lat
        lon = (f"{west:g}–{west + step_lon:g}° в.д." if west < 180
               else f"{360 - west - step_lon:g}–{360 - west:g}° з.д.")
        return f"{south:g}–{south + step_lat:g}° с.ш., {lon}"

    return index.ravel(), [label(c, r) for c, r in cells]
