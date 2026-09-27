"""
Тесты IDW-интерполяции CompactWeatherRepository.

Используют in-memory SQLite — не требуют реального архива.
Геометрия: шаг сетки 0.25°, центральный узел (44.50, 38.00),
соседи строго на расстоянии 0.25° — гарантированно попадают в _neighbourhood_3x3.
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.weather_history.archive.repository import (
    COORD_SCALE,
    EPS,
    GRID_STEP_DEG,
    VALUE_SCALE,
    CompactWeatherRepository,
    GridPoint,
    InterpolatedWeatherRecord,
    encode_coordinate,
    encode_day,
    encode_value,
)

# ── константы ──────────────────────────────────────────────────────────────

S = GRID_STEP_DEG           # 0.25°
CLAT = 44.50
CLON = 38.00
DATE1 = date(2010, 6, 1)
DATE2 = date(2010, 6, 2)


# ── вспомогательные функции ────────────────────────────────────────────────

def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE weather_grid_points (
            id    INTEGER PRIMARY KEY,
            lat_i INTEGER NOT NULL,
            lon_i INTEGER NOT NULL
        );
        CREATE TABLE weather_days (
            id                INTEGER PRIMARY KEY,
            point_id          INTEGER NOT NULL,
            day               INTEGER NOT NULL,
            wind_speed_max_i  INTEGER,
            wind_speed_mean_i INTEGER,
            wind_gust_max_i   INTEGER,
            wind_direction_i  INTEGER
        );
        CREATE UNIQUE INDEX uq_point_day ON weather_days(point_id, day);
    """)


def _add_point(con: sqlite3.Connection, pid: int, lat: float, lon: float) -> None:
    con.execute(
        "INSERT INTO weather_grid_points(id, lat_i, lon_i) VALUES (?,?,?)",
        (pid, encode_coordinate(lat), encode_coordinate(lon)),
    )


def _add_day(
    con: sqlite3.Connection,
    point_id: int,
    obs_date: date,
    ws_max: float | None = None,
    ws_mean: float | None = None,
    wg_max: float | None = None,
    wd: float | None = None,
) -> None:
    con.execute(
        """INSERT INTO weather_days
           (point_id, day, wind_speed_max_i, wind_speed_mean_i,
            wind_gust_max_i, wind_direction_i)
           VALUES (?,?,?,?,?,?)""",
        (
            point_id,
            encode_day(obs_date),
            encode_value(ws_max) if ws_max is not None else None,
            encode_value(ws_mean) if ws_mean is not None else None,
            encode_value(wg_max) if wg_max is not None else None,
            encode_value(wd) if wd is not None else None,
        ),
    )


def _make_repo(*setup_fns) -> CompactWeatherRepository:
    """Создаёт временную БД, применяет функции setup_fns, возвращает репо."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)

    con = sqlite3.connect(path)
    _create_schema(con)
    for fn in setup_fns:
        fn(con)
    con.commit()
    con.close()

    # CompactWeatherRepository открывает БД в режиме query_only,
    # поэтому создаём её заранее и передаём путь
    repo = CompactWeatherRepository.__new__(CompactWeatherRepository)
    repo._path = path
    repo._con = sqlite3.connect(path, check_same_thread=False)
    repo._con.row_factory = sqlite3.Row
    repo._con.execute("PRAGMA cache_size = -4096")
    return repo


# ── тесты ─────────────────────────────────────────────────────────────────


class TestNeighbourhood3x3:
    """_neighbourhood_3x3 находит все узлы в окрестности центра."""

    def test_full_nine_nodes(self):
        """Полная окрестность 3×3 — возвращает 9 узлов."""
        def setup(con):
            pid = 1
            for dlat in (-S, 0, S):
                for dlon in (-S, 0, S):
                    _add_point(con, pid, CLAT + dlat, CLON + dlon)
                    pid += 1

        repo = _make_repo(setup)
        pts = repo._neighbourhood_3x3(CLAT + 0.01, CLON + 0.01)
        assert len(pts) == 9

    def test_partial_neighbourhood(self):
        """Отсутствующие узлы не включаются в окрестность."""
        def setup(con):
            _add_point(con, 1, CLAT, CLON)          # центр
            _add_point(con, 2, CLAT, CLON + S)      # восток

        repo = _make_repo(setup)
        pts = repo._neighbourhood_3x3(CLAT + 0.01, CLON + 0.01)
        ids = {p.id for p in pts}
        assert ids == {1, 2}

    def test_centre_is_included(self):
        """Центральный узел всегда входит в окрестность."""
        def setup(con):
            _add_point(con, 1, CLAT, CLON)

        repo = _make_repo(setup)
        pts = repo._neighbourhood_3x3(CLAT, CLON)
        assert len(pts) == 1
        assert pts[0].lat == pytest.approx(CLAT)
        assert pts[0].lon == pytest.approx(CLON)


class TestExactNodeMatch:
    """При совпадении запроса с узлом возвращается запись узла напрямую."""

    def test_exact_match_returns_node_record(self):
        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_point(con, 2, CLAT, CLON + S)
            _add_day(con, 1, DATE1, ws_max=10.0, ws_mean=5.0, wg_max=20.0, wd=90.0)
            _add_day(con, 2, DATE1, ws_max=20.0, ws_mean=15.0, wg_max=40.0, wd=180.0)

        repo = _make_repo(setup)
        # запрос строго в узел 1
        records = repo.get_interpolated_timeseries(CLAT, CLON, DATE1, DATE1)
        assert len(records) == 1
        r = records[0]
        assert r.wind_speed_max == pytest.approx(10.0, abs=0.02)
        assert r.wind_direction == pytest.approx(90.0, abs=0.02)


class TestIdwScalar:
    """Скалярные поля интерполируются как IDW p=2."""

    def test_equidistant_two_nodes_gives_mean(self):
        """
        Два узла на равном расстоянии от запроса →
        IDW = среднее арифметическое.
        """
        lat_q = CLAT        # запрос строго между двумя узлами по долготе
        lon_q = CLON + S / 2  # посередине между CLON и CLON+S

        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_point(con, 2, CLAT, CLON + S)
            _add_day(con, 1, DATE1, ws_max=10.0)
            _add_day(con, 2, DATE1, ws_max=20.0)

        repo = _make_repo(setup)
        records = repo.get_interpolated_timeseries(lat_q, lon_q, DATE1, DATE1)
        assert len(records) == 1
        assert records[0].wind_speed_max == pytest.approx(15.0, abs=0.1)

    def test_closer_node_has_more_weight(self):
        """
        Узел ближе к запросу получает больший вес → IDW ≠ среднее.
        """
        # Запрос смещён на 0.1*S от центра в сторону узла 1
        lat_q = CLAT
        lon_q = CLON + S * 0.1

        def setup(con):
            _add_point(con, 1, CLAT, CLON)           # dist = 0.1*S
            _add_point(con, 2, CLAT, CLON + S)       # dist = 0.9*S
            _add_day(con, 1, DATE1, ws_max=10.0)
            _add_day(con, 2, DATE1, ws_max=20.0)

        repo = _make_repo(setup)
        records = repo.get_interpolated_timeseries(lat_q, lon_q, DATE1, DATE1)
        assert len(records) == 1
        val = records[0].wind_speed_max
        # IDW смещён к узлу 1 (10.0), результат должен быть < 15.0
        assert val < 15.0
        assert val > 10.0


class TestCircularDirectionIdw:
    """Направление ветра интерполируется через sin/cos."""

    def test_north_wrap_350_and_10(self):
        """
        350° и 10° равноудалённо → ожидаем 0° (≡ 360°), не 180°.
        """
        result = CompactWeatherRepository._idw_direction([(350.0, 1.0), (10.0, 1.0)])
        assert result is not None
        # Результирующий угол должен быть близок к 0° (или 360°)
        assert min(abs(result), abs(result - 360.0)) < 1.0

    def test_opposite_equal_weight_returns_none(self):
        """
        90° и 270° с равными весами → нулевой вектор → None.
        """
        result = CompactWeatherRepository._idw_direction([(90.0, 1.0), (270.0, 1.0)])
        assert result is None

    def test_single_direction_passthrough(self):
        """Одно направление возвращается без изменений."""
        result = CompactWeatherRepository._idw_direction([(135.0, 1.0)])
        assert result == pytest.approx(135.0, abs=0.01)

    def test_empty_returns_none(self):
        result = CompactWeatherRepository._idw_direction([])
        assert result is None

    def test_direction_interpolation_through_360(self):
        """
        Два узла: wd=355° и wd=5°, разные веса →
        результат должен быть в диапазоне [350°, 360°] ∪ [0°, 10°].
        """
        # w1 > w2 → результат ближе к 355°
        result = CompactWeatherRepository._idw_direction([(355.0, 2.0), (5.0, 1.0)])
        assert result is not None
        # Нормализуем в диапазон [-180, 180] относительно 0°
        diff = min(abs(result), abs(result - 360.0))
        assert diff < 10.0


class TestMissingFieldsHandling:
    """Отсутствующие поля обрабатываются независимо."""

    def test_partial_fields_per_node(self):
        """
        Узел 1 имеет только ws_max, узел 2 — только wd.
        Результат: ws_max из узла 1, wd из узла 2, остальные None.
        """
        lat_q = CLAT
        lon_q = CLON + S * 0.5  # равноудалённо

        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_point(con, 2, CLAT, CLON + S)
            _add_day(con, 1, DATE1, ws_max=12.0)
            _add_day(con, 2, DATE1, wd=270.0)

        repo = _make_repo(setup)
        records = repo.get_interpolated_timeseries(lat_q, lon_q, DATE1, DATE1)
        assert len(records) == 1
        r = records[0]
        assert r.wind_speed_max == pytest.approx(12.0, abs=0.02)
        assert r.wind_direction == pytest.approx(270.0, abs=0.02)
        assert r.wind_speed_mean is None
        assert r.wind_gust_max is None

    def test_no_data_for_day_returns_nothing(self):
        """День без данных ни у одного соседа → пустой список."""
        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_day(con, 1, DATE1, ws_max=5.0)
            # DATE2 — нет записей

        repo = _make_repo(setup)
        records = repo.get_interpolated_timeseries(CLAT + 0.01, CLON + 0.01, DATE2, DATE2)
        assert records == []


class TestDateFiltering:
    """Фильтрация по диапазону дат."""

    def test_start_end_date_filter(self):
        """Возвращаются только записи внутри [start_date, end_date]."""
        d0 = date(2010, 1, 1)
        d1 = date(2010, 1, 2)
        d2 = date(2010, 1, 3)

        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_day(con, 1, d0, ws_max=1.0)
            _add_day(con, 1, d1, ws_max=2.0)
            _add_day(con, 1, d2, ws_max=3.0)

        repo = _make_repo(setup)
        records = repo.get_interpolated_timeseries(CLAT, CLON, d0, d1)
        dates = {r.obs_date for r in records}
        assert d0 in dates
        assert d1 in dates
        assert d2 not in dates

    def test_empty_range_returns_empty(self):
        """Диапазон без данных → пустой список."""
        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_day(con, 1, DATE1, ws_max=5.0)

        repo = _make_repo(setup)
        future = date(2099, 1, 1)
        records = repo.get_interpolated_timeseries(CLAT, CLON, future, future)
        assert records == []


class TestInterpolatedRecordFields:
    """InterpolatedWeatherRecord содержит координаты точки запроса."""

    def test_query_coords_stored(self):
        lat_q = CLAT + 0.05
        lon_q = CLON + 0.05

        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_day(con, 1, DATE1, ws_max=8.0, wd=45.0)

        repo = _make_repo(setup)
        records = repo.get_interpolated_timeseries(lat_q, lon_q, DATE1, DATE1)
        assert len(records) == 1
        assert records[0].query_lat == pytest.approx(lat_q)
        assert records[0].query_lon == pytest.approx(lon_q)
        assert records[0].obs_date == DATE1


class TestGetInterpolationNeighbours:
    """Публичный диагностический метод get_interpolation_neighbours."""

    def test_returns_dist_and_weight(self):
        def setup(con):
            _add_point(con, 1, CLAT, CLON)
            _add_point(con, 2, CLAT + S, CLON)

        repo = _make_repo(setup)
        result = repo.get_interpolation_neighbours(CLAT + 0.05, CLON)
        assert len(result) >= 1
        pt, dist, w = result[0]
        assert isinstance(pt, GridPoint)
        assert dist >= 0.0
        assert w > 0.0

    def test_exact_match_returns_inf_weight(self):
        """При совпадении координат вес = inf."""
        def setup(con):
            _add_point(con, 1, CLAT, CLON)

        repo = _make_repo(setup)
        result = repo.get_interpolation_neighbours(CLAT, CLON)
        assert len(result) == 1
        _, _, w = result[0]
        assert math.isinf(w)
