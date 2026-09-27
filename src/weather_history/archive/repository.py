"""
Интерфейс чтения компактного архива метеоданных.

Семантика:
- В archive weather_grid_points хранятся узлы регулярной сетки модели.
- Для произвольной точки (lat, lon) ищется ближайший узел сетки.
- Метод containing_grid_cell() оставлен только как алиас совместимости и
  возвращает ближайший узел.
- Метод get_timeseries_for_point() поддерживает два режима:
    1) новый:  (point: GridPoint, start_date=None, end_date=None)
    2) старый: (lat: float, lon: float, start_date=None, end_date=None)
- Метод get_interpolated_timeseries() выполняет IDW-интерполяцию по
  окрестности 3×3 узлов сетки (центральный + до 8 соседей).
  Направление ветра интерполируется через sin/cos (корректная обработка
  перехода через 0°/360°). Скалярные поля — прямое взвешенное среднее.
  Если расстояние до узла < EPS — значение возвращается напрямую.
  Дни, за которые нет данных ни у одного соседа, пропускаются.
  Нулевой результирующий вектор направления (равновесные противоположные
  направления) возвращает None.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


EPOCH_DATE = date(1940, 1, 1)
VALUE_SCALE = 100
COORD_SCALE = 1000
GRID_STEP_DEG = 0.25
HALF_GRID_STEP_DEG = GRID_STEP_DEG / 2.0
EPS = 1e-9

# IDW степень весов (p=2 — стандарт)
_IDW_POWER = 2


def encode_day(value: date) -> int:
    return (value - EPOCH_DATE).days


def decode_day(value: int) -> date:
    return EPOCH_DATE + timedelta(days=value)


def encode_value(value: float) -> int:
    return int(round(value * VALUE_SCALE))


def decode_value(value: int | None) -> float | None:
    if value is None:
        return None
    return value / VALUE_SCALE


def encode_coordinate(value: float) -> int:
    return int(round(value * COORD_SCALE))


def decode_coordinate(value: int) -> float:
    return value / COORD_SCALE


def _coerce_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@dataclass(frozen=True, slots=True)
class GridPoint:
    id: int
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class WeatherRecord:
    point: GridPoint
    obs_date: date
    wind_speed_max: float | None
    wind_speed_mean: float | None
    wind_gust_max: float | None
    wind_direction: float | None


@dataclass(frozen=True, slots=True)
class InterpolatedWeatherRecord:
    """Результат IDW-интерполяции; значение относится к произвольной
    точке (query_lat, query_lon), а не к узлу сетки."""

    obs_date: date
    query_lat: float
    query_lon: float
    wind_speed_max: float | None
    wind_speed_mean: float | None
    wind_gust_max: float | None
    wind_direction: float | None


_EMPTY_DF_COLUMNS = [
    "point_id", "point_lat", "point_lon",
    "wind_speed_max", "wind_speed_mean",
    "wind_gust_max", "wind_direction",
]


class CompactWeatherRepository:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser().resolve()
        self._con = self._open()

    def __enter__(self) -> "CompactWeatherRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._con.close()

    def _open(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        con.execute("PRAGMA cache_size = -65536")
        con.execute("PRAGMA mmap_size = 536870912")
        return con

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Узлы сетки
    # ------------------------------------------------------------------

    def list_grid_points(self) -> list[GridPoint]:
        rows = self._con.execute(
            "SELECT id, lat_i, lon_i FROM weather_grid_points ORDER BY lat_i, lon_i"
        ).fetchall()
        return [
            GridPoint(
                id=row["id"],
                lat=decode_coordinate(row["lat_i"]),
                lon=decode_coordinate(row["lon_i"]),
            )
            for row in rows
        ]

    def get_point_by_id(self, point_id: int) -> GridPoint | None:
        row = self._con.execute(
            "SELECT id, lat_i, lon_i FROM weather_grid_points WHERE id = ?",
            (point_id,),
        ).fetchone()
        if row is None:
            return None
        return GridPoint(
            id=row["id"],
            lat=decode_coordinate(row["lat_i"]),
            lon=decode_coordinate(row["lon_i"]),
        )

    def nearest_grid_point(self, lat: float, lon: float) -> GridPoint | None:
        lat_i = encode_coordinate(lat)
        lon_i = encode_coordinate(lon)
        row = self._con.execute(
            """
            SELECT
                id, lat_i, lon_i,
                ((lat_i - ?) * (lat_i - ?) + (lon_i - ?) * (lon_i - ?)) AS dist2
            FROM weather_grid_points
            ORDER BY dist2 ASC, id ASC
            LIMIT 1
            """,
            (lat_i, lat_i, lon_i, lon_i),
        ).fetchone()
        if row is None:
            return None
        return GridPoint(
            id=row["id"],
            lat=decode_coordinate(row["lat_i"]),
            lon=decode_coordinate(row["lon_i"]),
        )

    def containing_grid_cell(self, lat: float, lon: float) -> GridPoint | None:
        """
        Возвращает узел сетки только если точка попадает в его ячейку
        (квадрат половинного шага вокруг узла). Если точка вне ячейки — None.
        """
        point = self.nearest_grid_point(lat, lon)
        if point is None:
            return None
        if (
            abs(lat - point.lat) <= HALF_GRID_STEP_DEG + EPS
            and abs(lon - point.lon) <= HALF_GRID_STEP_DEG + EPS
        ):
            return point
        return None

    def _fetch_neighbour(self, clat: float, clon: float) -> GridPoint | None:
        """Возвращает GridPoint строго по координатам узла сетки или None."""
        lat_i = encode_coordinate(round(clat, 6))
        lon_i = encode_coordinate(round(clon, 6))
        row = self._con.execute(
            "SELECT id, lat_i, lon_i FROM weather_grid_points "
            "WHERE lat_i = ? AND lon_i = ?",
            (lat_i, lon_i),
        ).fetchone()
        if row is None:
            return None
        return GridPoint(
            id=row["id"],
            lat=decode_coordinate(row["lat_i"]),
            lon=decode_coordinate(row["lon_i"]),
        )

    def _neighbourhood_3x3(self, lat: float, lon: float) -> list[GridPoint]:
        """
        Возвращает список узлов сетки в окрестности 3×3 вокруг
        ближайшего к (lat, lon) узла. Отсутствующие узлы пропускаются.
        """
        centre = self.nearest_grid_point(lat, lon)
        if centre is None:
            return []
        points: list[GridPoint] = []
        for dlat in (-GRID_STEP_DEG, 0.0, GRID_STEP_DEG):
            for dlon in (-GRID_STEP_DEG, 0.0, GRID_STEP_DEG):
                candidate = self._fetch_neighbour(
                    centre.lat + dlat, centre.lon + dlon
                )
                if candidate is not None:
                    points.append(candidate)
        return points

    def get_interpolation_neighbours(
        self, lat: float, lon: float
    ) -> list[tuple[GridPoint, float, float]]:
        """
        Публичный диагностический метод.
        Возвращает список (GridPoint, dist_deg, idw_weight) для окрестности
        3×3 вокруг точки (lat, lon).
        """
        neighbours = self._neighbourhood_3x3(lat, lon)
        result = []
        for pt in neighbours:
            dist = math.sqrt((lat - pt.lat) ** 2 + (lon - pt.lon) ** 2)
            w = 1.0 / (dist ** _IDW_POWER) if dist > EPS else float("inf")
            result.append((pt, dist, w))
        return result

    # ------------------------------------------------------------------
    # Одиночные записи
    # ------------------------------------------------------------------

    def get_record_for_point(
        self,
        point: GridPoint,
        obs_date: date | str,
    ) -> WeatherRecord | None:
        obs_date = _coerce_date(obs_date)
        assert obs_date is not None
        row = self._con.execute(
            """
            SELECT day, wind_speed_max_i, wind_speed_mean_i,
                   wind_gust_max_i, wind_direction_i
            FROM weather_days
            WHERE point_id = ? AND day = ?
            """,
            (point.id, encode_day(obs_date)),
        ).fetchone()
        if row is None:
            return None
        return WeatherRecord(
            point=point,
            obs_date=decode_day(row["day"]),
            wind_speed_max=decode_value(row["wind_speed_max_i"]),
            wind_speed_mean=decode_value(row["wind_speed_mean_i"]),
            wind_gust_max=decode_value(row["wind_gust_max_i"]),
            wind_direction=decode_value(row["wind_direction_i"]),
        )

    def get_record(
        self,
        lat: float,
        lon: float,
        obs_date: date | str,
    ) -> WeatherRecord | None:
        point = self.nearest_grid_point(lat, lon)
        if point is None:
            return None
        return self.get_record_for_point(point, obs_date)

    def get_nearest_point_record(
        self,
        lat: float,
        lon: float,
        obs_date: date | str,
    ) -> tuple[GridPoint | None, WeatherRecord | None]:
        point = self.nearest_grid_point(lat, lon)
        if point is None:
            return None, None
        return point, self.get_record_for_point(point, obs_date)

    # ------------------------------------------------------------------
    # Временные ряды (без интерполяции)
    # ------------------------------------------------------------------

    def _get_timeseries_for_grid_point(
        self,
        point: GridPoint,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> list[WeatherRecord]:
        start_date = _coerce_date(start_date)
        end_date = _coerce_date(end_date)

        conditions = ["point_id = ?"]
        params: list[Any] = [point.id]

        if start_date is not None:
            conditions.append("day >= ?")
            params.append(encode_day(start_date))
        if end_date is not None:
            conditions.append("day <= ?")
            params.append(encode_day(end_date))

        sql = f"""
            SELECT day, wind_speed_max_i, wind_speed_mean_i,
                   wind_gust_max_i, wind_direction_i
            FROM weather_days
            WHERE {" AND ".join(conditions)}
            ORDER BY day
        """
        rows = self._con.execute(sql, tuple(params)).fetchall()
        return [
            WeatherRecord(
                point=point,
                obs_date=decode_day(row["day"]),
                wind_speed_max=decode_value(row["wind_speed_max_i"]),
                wind_speed_mean=decode_value(row["wind_speed_mean_i"]),
                wind_gust_max=decode_value(row["wind_gust_max_i"]),
                wind_direction=decode_value(row["wind_direction_i"]),
            )
            for row in rows
        ]

    def get_timeseries_for_point(self, *args) -> list[WeatherRecord]:
        if not args:
            raise TypeError("get_timeseries_for_point() requires arguments")

        first = args[0]

        if isinstance(first, GridPoint):
            point = first
            start_date = args[1] if len(args) >= 2 else None
            end_date = args[2] if len(args) >= 3 else None
            return self._get_timeseries_for_grid_point(point, start_date, end_date)

        if len(args) < 2:
            raise TypeError(
                "get_timeseries_for_point(lat, lon, start_date=None, end_date=None)"
            )

        lat = float(args[0])
        lon = float(args[1])
        start_date = args[2] if len(args) >= 3 else None
        end_date = args[3] if len(args) >= 4 else None

        point = self.nearest_grid_point(lat, lon)
        if point is None:
            return []
        return self._get_timeseries_for_grid_point(point, start_date, end_date)

    def get_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> list[WeatherRecord]:
        point = self.nearest_grid_point(lat, lon)
        if point is None:
            return []
        return self._get_timeseries_for_grid_point(point, start_date, end_date)

    # ------------------------------------------------------------------
    # IDW-интерполяция по окрестности 3×3
    # ------------------------------------------------------------------

    @staticmethod
    def _idw_scalar(
        values_weights: list[tuple[float, float]],
    ) -> float | None:
        """Взвешенное среднее IDW для скалярного поля."""
        if not values_weights:
            return None
        num = sum(v * w for v, w in values_weights)
        den = sum(w for _, w in values_weights)
        return num / den if den > EPS else None

    @staticmethod
    def _idw_direction(
        dirs_weights: list[tuple[float, float]],
    ) -> float | None:
        """
        IDW для направления ветра через декомпозицию на sin/cos.
        Возвращает угол в градусах [0, 360) или None если результирующий
        вектор нулевой (равновесные противоположные направления).
        """
        if not dirs_weights:
            return None
        sin_sum = sum(math.sin(math.radians(d)) * w for d, w in dirs_weights)
        cos_sum = sum(math.cos(math.radians(d)) * w for d, w in dirs_weights)
        den = sum(w for _, w in dirs_weights)
        if den < EPS:
            return None
        sin_mean = sin_sum / den
        cos_mean = cos_sum / den
        # Нулевой результирующий вектор — направление не определено
        if math.hypot(sin_mean, cos_mean) < EPS:
            return None
        return math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0

    def _interpolate_day(
        self,
        neighbours: list[GridPoint],
        lat: float,
        lon: float,
        day_int: int,
    ) -> InterpolatedWeatherRecord | None:
        """
        IDW-интерполяция для одного дня по списку соседних узлов.
        Возвращает None если ни у одного узла нет данных за этот день.
        """
        obs_date = decode_day(day_int)

        speed_max_wv: list[tuple[float, float]] = []
        speed_mean_wv: list[tuple[float, float]] = []
        gust_max_wv: list[tuple[float, float]] = []
        direction_wv: list[tuple[float, float]] = []

        for pt in neighbours:
            dist = math.sqrt((lat - pt.lat) ** 2 + (lon - pt.lon) ** 2)

            if dist < EPS:
                # точка совпадает с узлом — возвращаем напрямую
                rec = self.get_record_for_point(pt, obs_date)
                if rec is None:
                    continue
                return InterpolatedWeatherRecord(
                    obs_date=obs_date,
                    query_lat=lat,
                    query_lon=lon,
                    wind_speed_max=rec.wind_speed_max,
                    wind_speed_mean=rec.wind_speed_mean,
                    wind_gust_max=rec.wind_gust_max,
                    wind_direction=rec.wind_direction,
                )

            rec = self.get_record_for_point(pt, obs_date)
            if rec is None:
                continue

            w = 1.0 / (dist ** _IDW_POWER)

            if rec.wind_speed_max is not None:
                speed_max_wv.append((rec.wind_speed_max, w))
            if rec.wind_speed_mean is not None:
                speed_mean_wv.append((rec.wind_speed_mean, w))
            if rec.wind_gust_max is not None:
                gust_max_wv.append((rec.wind_gust_max, w))
            if rec.wind_direction is not None:
                direction_wv.append((rec.wind_direction, w))

        if not any([speed_max_wv, speed_mean_wv, gust_max_wv, direction_wv]):
            return None

        return InterpolatedWeatherRecord(
            obs_date=obs_date,
            query_lat=lat,
            query_lon=lon,
            wind_speed_max=self._idw_scalar(speed_max_wv),
            wind_speed_mean=self._idw_scalar(speed_mean_wv),
            wind_gust_max=self._idw_scalar(gust_max_wv),
            wind_direction=self._idw_direction(direction_wv),
        )

    def _get_day_range(
        self,
        neighbours: list[GridPoint],
        start_date: date | None,
        end_date: date | None,
    ) -> list[int]:
        """
        Возвращает отсортированный список уникальных дней (int) из таблицы
        weather_days для заданных point_id с учётом фильтров по дате.
        """
        if not neighbours:
            return []

        ids = tuple(pt.id for pt in neighbours)
        placeholders = ",".join("?" * len(ids))
        conditions = [f"point_id IN ({placeholders})"]
        params: list[Any] = list(ids)

        if start_date is not None:
            conditions.append("day >= ?")
            params.append(encode_day(start_date))
        if end_date is not None:
            conditions.append("day <= ?")
            params.append(encode_day(end_date))

        sql = f"""
            SELECT DISTINCT day
            FROM weather_days
            WHERE {" AND ".join(conditions)}
            ORDER BY day
        """
        rows = self._con.execute(sql, tuple(params)).fetchall()
        return [row["day"] for row in rows]

    def get_interpolated_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> list[InterpolatedWeatherRecord]:
        """
        Возвращает IDW-интерполированный временной ряд для произвольной
        точки (lat, lon) по окрестности 3×3 узлов сетки.

        Направление ветра интерполируется через sin/cos.
        Скалярные поля — прямое взвешенное среднее по расстоянию^2.
        Дни, за которые нет данных ни у одного соседнего узла, пропускаются.
        Нулевой результирующий вектор направления возвращает None.
        """
        start_date = _coerce_date(start_date)
        end_date = _coerce_date(end_date)

        neighbours = self._neighbourhood_3x3(lat, lon)
        if not neighbours:
            return []

        day_ints = self._get_day_range(neighbours, start_date, end_date)

        result: list[InterpolatedWeatherRecord] = []
        for day_int in day_ints:
            rec = self._interpolate_day(neighbours, lat, lon, day_int)
            if rec is not None:
                result.append(rec)

        return result

    def get_interpolated_dataframe(
        self,
        lat: float,
        lon: float,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ):
        """
        Возвращает IDW-интерполированный ряд как DataFrame с индексом obs_date.
        Требует pandas.
        """
        if pd is None:
            raise ImportError("pandas не установлен")

        records = self.get_interpolated_timeseries(lat, lon, start_date, end_date)

        if not records:
            return pd.DataFrame(
                columns=[
                    "query_lat", "query_lon",
                    "wind_speed_max", "wind_speed_mean",
                    "wind_gust_max", "wind_direction",
                ]
            )

        df = pd.DataFrame(
            {
                "obs_date": [r.obs_date for r in records],
                "query_lat": [r.query_lat for r in records],
                "query_lon": [r.query_lon for r in records],
                "wind_speed_max": [r.wind_speed_max for r in records],
                "wind_speed_mean": [r.wind_speed_mean for r in records],
                "wind_gust_max": [r.wind_gust_max for r in records],
                "wind_direction": [r.wind_direction for r in records],
            }
        )
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        return df.set_index("obs_date")

    # ------------------------------------------------------------------
    # DataFrame-обёртки (без интерполяции)
    # ------------------------------------------------------------------

    def get_dataframe_for_point(
        self,
        point: GridPoint,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ):
        if pd is None:
            raise ImportError("pandas не установлен")

        rows = self._get_timeseries_for_grid_point(point, start_date, end_date)
        if not rows:
            return pd.DataFrame(columns=_EMPTY_DF_COLUMNS)

        df = pd.DataFrame(
            {
                "obs_date": [r.obs_date for r in rows],
                "point_id": [r.point.id for r in rows],
                "point_lat": [r.point.lat for r in rows],
                "point_lon": [r.point.lon for r in rows],
                "wind_speed_max": [r.wind_speed_max for r in rows],
                "wind_speed_mean": [r.wind_speed_mean for r in rows],
                "wind_gust_max": [r.wind_gust_max for r in rows],
                "wind_direction": [r.wind_direction for r in rows],
            }
        )
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        return df.set_index("obs_date")

    def get_dataframe(
        self,
        lat: float,
        lon: float,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ):
        if pd is None:
            raise ImportError("pandas не установлен")

        point = self.nearest_grid_point(lat, lon)
        if point is None:
            return pd.DataFrame(columns=_EMPTY_DF_COLUMNS)

        return self.get_dataframe_for_point(point, start_date, end_date)

    # ------------------------------------------------------------------
    # Информация об архиве
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        point_count = self._con.execute(
            "SELECT COUNT(*) AS value FROM weather_grid_points"
        ).fetchone()["value"]

        row = self._con.execute(
            """
            SELECT
                COUNT(*) AS day_count,
                MIN(day) AS day_min,
                MAX(day) AS day_max
            FROM weather_days
            """
        ).fetchone()

        result: dict[str, Any] = {
            "db_path": str(self._path),
            "grid_points": point_count,
            "weather_days": row["day_count"],
            "date_min": decode_day(row["day_min"]) if row["day_min"] is not None else None,
            "date_max": decode_day(row["day_max"]) if row["day_max"] is not None else None,
        }
        result["point_count"] = result["grid_points"]
        result["day_count"] = result["weather_days"]
        return result
