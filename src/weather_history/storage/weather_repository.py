"""
src/weather_history/storage/weather_repository.py

Единая точка доступа к данным о ветре через БД:
- скачивание из Open-Meteo с проверкой "что уже скачано" (не дублирует запросы);
- сохранение скорости И направления ветра в одной строке WeatherDayModel;
- источник данных (модель реанализа) и единицы измерения вынесены в
  справочник WeatherSourceModel и не копируются в каждую суточную запись;
- чтение временных рядов по узлу сетки, по cell_id или по произвольной точке.

Порывы ветра (wind_gusts_10m_max) не запрашиваются и не хранятся.

Работает на SQLite и PostgreSQL без изменения кода — только через DATABASE_URL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.weather_history.storage.db import init_db, session_scope
from src.weather_history.storage.models import (
    DownloadSegmentModel,
    GridPointModel,
    WeatherDayModel,
    WeatherSourceModel,
)
from src.weather_history.wheather_downloaders.open_meteo.GeoJsonGridBuilder import (
    GeoJsonGridBuilder,
)
from src.weather_history.wheather_downloaders.open_meteo.OpenMeteoArchiveClient import (
    OpenMeteoArchiveClient,
)
from src.weather_history.wheather_downloaders.open_meteo.models import GridPoint

# ──────────────────────────────────────────────────────────────────────────
# Скорость и направление ветра. Порывы (wind_gusts_10m_max) не запрашиваются.
# ──────────────────────────────────────────────────────────────────────────
WIND_SPEED_VAR = "wind_speed_10m_max"
WIND_DIRECTION_VAR = "wind_direction_10m_dominant"
DAILY_VARIABLES: tuple[str, ...] = (WIND_SPEED_VAR, WIND_DIRECTION_VAR)


@dataclass(frozen=True)
class WeatherDownloadSettings:
    model: str = "era5"
    grid_step: float = 0.25
    grid_center_offset: float = 0.125
    cover_points_with_cells: bool = True
    extra_border_cells: int = 1
    timezone: str = "GMT"
    cell_selection: str = "nearest"
    batch_size: int = 25
    request_pause_seconds: float = 0.0
    user_agent: str = "BreachTheBeach/0.1.0"
    archive_min_date: str = "1940-01-01"
    archive_lag_days: int = 7


class WeatherRepository:
    """Скачивание и чтение данных о ветре через БД (SQLAlchemy)."""

    def __init__(self, settings: WeatherDownloadSettings | None = None) -> None:
        self.settings = settings or WeatherDownloadSettings()
        self.grid_builder = GeoJsonGridBuilder(
            grid_step=self.settings.grid_step,
            grid_center_offset=self.settings.grid_center_offset,
            cover_points_with_cells=self.settings.cover_points_with_cells,
            extra_border_cells=self.settings.extra_border_cells,
        )
        self.client = OpenMeteoArchiveClient(user_agent=self.settings.user_agent)
        init_db()
        self._source_id_cache: dict[tuple[str, str | None, str | None], int] = {}

    # ────────────────────────────────────────────────────────────────────
    # СКАЧИВАНИЕ ПО ГЕОМЕТРИИ (строит сетку сам)
    # ────────────────────────────────────────────────────────────────────

    def download_from_geojson(
        self,
        geojson_path: str | Path,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Строит сетку по геометрии из geojson_path, докачивает недостающие
        даты для каждого узла и сохраняет всё в БД.
        """
        normalized_start, normalized_end = self._normalize_requested_range(start_date, end_date)

        source_bbox, weather_bbox, grid_points = self.grid_builder.build_grid(geojson_path)
        logger.info(f"Source bbox: {source_bbox}")
        logger.info(f"Weather bbox: {weather_bbox}")
        logger.info(f"Grid points generated: {len(grid_points)}")

        return self._download_for_points(
            grid_points=grid_points,
            start_date=normalized_start,
            end_date=normalized_end,
            extra_result={"source_bbox": source_bbox, "weather_bbox": weather_bbox},
        )

    # ────────────────────────────────────────────────────────────────────
    # СКАЧИВАНИЕ ПО ГОТОВЫМ ТОЧКАМ (используется WeatherPointsLoader)
    # ────────────────────────────────────────────────────────────────────

    def download_for_points(
        self,
        grid_points: list[GridPoint],
        start_date: str,
        end_date: str,
        cell_ids: list[str | None] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Скачивает данные для уже готового списка узлов сетки
        (например, полученных из geojson-файла с точками ERA5).

        cell_ids — необязательный список идентификаторов ячеек (тот же
        порядок, что и grid_points), сохраняется в GridPointModel.cell_id.

        progress_callback(done: int, total: int) вызывается после обработки
        каждой задачи скачивания — используется для tqdm в вызывающем коде.
        """
        normalized_start, normalized_end = self._normalize_requested_range(start_date, end_date)
        return self._download_for_points(
            grid_points=grid_points,
            start_date=normalized_start,
            end_date=normalized_end,
            cell_ids=cell_ids,
            progress_callback=progress_callback,
        )

    def _download_for_points(
        self,
        grid_points: list[GridPoint],
        start_date: str,
        end_date: str,
        extra_result: dict | None = None,
        cell_ids: list[str | None] | None = None,
        progress_callback=None,
    ) -> dict:
        if cell_ids is None:
            cell_ids = [None] * len(grid_points)
        grid_point_ids = [
            self._get_or_create_grid_point(point, cell_id=cell_id)
            for point, cell_id in zip(grid_points, cell_ids, strict=True)
        ]

        # Источник данных (модель + единицы измерения) на момент проверки
        # кэша ещё не известен точно (единицы приходят в ответе Open-Meteo),
        # поэтому проверка "что уже скачано" смотрит на данные текущей модели
        # независимо от конкретной записи единиц измерения.
        point_missing_ranges: dict[int, list[tuple[str, str]]] = {}
        for point, grid_point_id in zip(grid_points, grid_point_ids, strict=True):
            missing = self._get_missing_ranges(
                grid_point_id=grid_point_id,
                start_date=start_date,
                end_date=end_date,
            )
            if missing:
                point_missing_ranges[grid_point_id] = missing

        logger.info(
            f"Points fully cached: {len(grid_points) - len(point_missing_ranges)}, "
            f"points needing download: {len(point_missing_ranges)}"
        )

        tasks = self._build_download_tasks(grid_points, grid_point_ids, point_missing_ranges)
        grouped_tasks = self._group_tasks_by_date_range(tasks)

        total_tasks = len(tasks)
        done_tasks = 0
        downloaded_segments = 0

        for (batch_start, batch_end), points_for_range in grouped_tasks.items():
            for batch in self._batched(points_for_range, self.settings.batch_size):
                batch_points = [item["point"] for item in batch]
                batch_ids = [item["grid_point_id"] for item in batch]

                payload, source_url = self.client.fetch(
                    points=batch_points,
                    start_date=batch_start,
                    end_date=batch_end,
                    daily_variables=DAILY_VARIABLES,
                    model=self.settings.model,
                    timezone=self.settings.timezone,
                    cell_selection=self.settings.cell_selection,
                )

                records = payload if isinstance(payload, list) else [payload]
                if len(records) != len(batch_points):
                    raise ValueError(
                        f"Response size mismatch: received {len(records)} records "
                        f"for {len(batch_points)} requested points"
                    )

                for grid_point_id, record in zip(batch_ids, records, strict=True):
                    self._save_segment(
                        grid_point_id=grid_point_id,
                        start_date=batch_start,
                        end_date=batch_end,
                        payload=record,
                        source_url=source_url,
                    )
                    downloaded_segments += 1
                    done_tasks += 1
                    if progress_callback is not None:
                        progress_callback(done_tasks, total_tasks)

                if self.settings.request_pause_seconds > 0:
                    time.sleep(self.settings.request_pause_seconds)

        logger.success(
            f"Download finished: {downloaded_segments} segments saved, "
            f"period {start_date}..{end_date}"
        )

        result = {
            "grid_points_count": len(grid_points),
            "downloaded_segments": downloaded_segments,
            "start_date": start_date,
            "end_date": end_date,
        }
        if extra_result:
            result.update(extra_result)
        return result

    # ────────────────────────────────────────────────────────────────────
    # ЧТЕНИЕ
    # ────────────────────────────────────────────────────────────────────

    def get_timeseries_for_point(
        self,
        req_lat: float,
        req_lon: float,
        start_date: str,
        end_date: str,
        model: str | None = None,
    ) -> list[dict]:
        """Временной ряд скорости и направления ветра для узла сетки по координатам."""
        model = model or self.settings.model
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        with session_scope() as session:
            grid_point = session.execute(
                select(GridPointModel).where(
                    GridPointModel.req_lat == req_lat,
                    GridPointModel.req_lon == req_lon,
                )
            ).scalar_one_or_none()

            if grid_point is None:
                raise KeyError(f"Grid point not found: lat={req_lat}, lon={req_lon}")

            return self._read_days(session, grid_point.id, model, start, end)

    def get_nearest_point_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        model: str | None = None,
    ) -> tuple[dict, list[dict]]:
        """
        Находит ближайший к заданной точке узел сетки (простая евклидова
        близость в градусах — достаточно для регулярной сетки 0.25°)
        и возвращает его временной ряд скорости и направления ветра.
        """
        model = model or self.settings.model
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        with session_scope() as session:
            grid_points = session.execute(select(GridPointModel)).scalars().all()
            if not grid_points:
                raise RuntimeError("No grid points in database. Run download first.")

            nearest = min(
                grid_points,
                key=lambda gp: (gp.req_lat - lat) ** 2 + (gp.req_lon - lon) ** 2,
            )

            point_info = {
                "cell_id": nearest.cell_id,
                "req_lat": nearest.req_lat,
                "req_lon": nearest.req_lon,
                "resolved_lat": nearest.resolved_lat,
                "resolved_lon": nearest.resolved_lon,
            }
            timeseries = self._read_days(session, nearest.id, model, start, end)
            return point_info, timeseries

    def get_timeseries_by_cell_id(
        self,
        cell_id: str,
        start_date: str,
        end_date: str,
        model: str | None = None,
    ) -> list[dict]:
        """Временной ряд скорости и направления ветра по cell_id из файла точек."""
        model = model or self.settings.model
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        with session_scope() as session:
            grid_point = session.execute(
                select(GridPointModel).where(GridPointModel.cell_id == cell_id)
            ).scalar_one_or_none()

            if grid_point is None:
                raise KeyError(f"Grid point not found: cell_id={cell_id}")

            return self._read_days(session, grid_point.id, model, start, end)

    @staticmethod
    def _read_days(session, grid_point_id: int, model: str, start: date, end: date) -> list[dict]:
        rows = session.execute(
            select(WeatherDayModel, WeatherSourceModel)
            .join(WeatherSourceModel, WeatherDayModel.source_id == WeatherSourceModel.id)
            .where(
                WeatherDayModel.grid_point_id == grid_point_id,
                WeatherSourceModel.model == model,
                WeatherDayModel.obs_date >= start,
                WeatherDayModel.obs_date <= end,
            )
            .order_by(WeatherDayModel.obs_date)
        ).all()

        return [
            {
                "date": day.obs_date.isoformat(),
                "wind_speed_max": day.wind_speed_max,
                "wind_direction": day.wind_direction,
                "model": source.model,
                "ws_unit": source.ws_unit,
                "wd_unit": source.wd_unit,
            }
            for day, source in rows
        ]

    def list_grid_points(self) -> list[dict]:
        with session_scope() as session:
            rows = session.execute(select(GridPointModel)).scalars().all()
            return [
                {
                    "id": row.id,
                    "cell_id": row.cell_id,
                    "req_lat": row.req_lat,
                    "req_lon": row.req_lon,
                    "ring_y": row.ring_y,
                    "ring_x": row.ring_x,
                    "resolved_lat": row.resolved_lat,
                    "resolved_lon": row.resolved_lon,
                }
                for row in rows
            ]

    def list_sources(self) -> list[dict]:
        """Справочник источников данных (модель + единицы измерения), уже сохранённых в БД."""
        with session_scope() as session:
            rows = session.execute(select(WeatherSourceModel)).scalars().all()
            return [
                {
                    "id": row.id,
                    "model": row.model,
                    "ws_unit": row.ws_unit,
                    "wd_unit": row.wd_unit,
                }
                for row in rows
            ]

    # ────────────────────────────────────────────────────────────────────
    # ВНУТРЕННЯЯ ЛОГИКА: справочник источников данных
    # ────────────────────────────────────────────────────────────────────

    def _get_or_create_source_id(
        self,
        model: str,
        ws_unit: str | None,
        wd_unit: str | None,
    ) -> int:
        cache_key = (model, ws_unit, wd_unit)
        if cache_key in self._source_id_cache:
            return self._source_id_cache[cache_key]

        with session_scope() as session:
            existing = session.execute(
                select(WeatherSourceModel).where(
                    WeatherSourceModel.model == model,
                    WeatherSourceModel.ws_unit == ws_unit,
                    WeatherSourceModel.wd_unit == wd_unit,
                )
            ).scalar_one_or_none()

            if existing is not None:
                self._source_id_cache[cache_key] = existing.id
                return existing.id

            new_source = WeatherSourceModel(model=model, ws_unit=ws_unit, wd_unit=wd_unit)
            session.add(new_source)
            try:
                session.flush()
                self._source_id_cache[cache_key] = new_source.id
                return new_source.id
            except IntegrityError:
                session.rollback()
                existing = session.execute(
                    select(WeatherSourceModel).where(
                        WeatherSourceModel.model == model,
                        WeatherSourceModel.ws_unit == ws_unit,
                        WeatherSourceModel.wd_unit == wd_unit,
                    )
                ).scalar_one()
                self._source_id_cache[cache_key] = existing.id
                return existing.id

    # ────────────────────────────────────────────────────────────────────
    # ВНУТРЕННЯЯ ЛОГИКА: узлы сетки
    # ────────────────────────────────────────────────────────────────────

    def _get_or_create_grid_point(self, point: GridPoint, cell_id: str | None = None) -> int:
        with session_scope() as session:
            existing = session.execute(
                select(GridPointModel).where(
                    GridPointModel.req_lat == point.lat,
                    GridPointModel.req_lon == point.lon,
                    GridPointModel.ring_y == point.ring_y,
                    GridPointModel.ring_x == point.ring_x,
                )
            ).scalar_one_or_none()

            if existing is not None:
                if cell_id and not existing.cell_id:
                    existing.cell_id = cell_id
                return existing.id

            new_point = GridPointModel(
                req_lat=point.lat,
                req_lon=point.lon,
                ring_y=point.ring_y,
                ring_x=point.ring_x,
                cell_id=cell_id,
            )
            session.add(new_point)
            try:
                session.flush()
                return new_point.id
            except IntegrityError:
                session.rollback()
                existing = session.execute(
                    select(GridPointModel).where(
                        GridPointModel.req_lat == point.lat,
                        GridPointModel.req_lon == point.lon,
                        GridPointModel.ring_y == point.ring_y,
                        GridPointModel.ring_x == point.ring_x,
                    )
                ).scalar_one()
                return existing.id

    # ────────────────────────────────────────────────────────────────────
    # ВНУТРЕННЯЯ ЛОГИКА: проверка что уже скачано
    # ────────────────────────────────────────────────────────────────────

    def _get_missing_ranges(
        self,
        grid_point_id: int,
        start_date: str,
        end_date: str,
    ) -> list[tuple[str, str]]:
        requested_dates = self._date_set(start_date, end_date)
        cached_dates = self._collect_cached_dates(grid_point_id, start_date, end_date)
        missing_dates = sorted(requested_dates - cached_dates)
        return self._dates_to_ranges(missing_dates)

    def _collect_cached_dates(
        self,
        grid_point_id: int,
        start_date: str,
        end_date: str,
    ) -> set[date]:
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        with session_scope() as session:
            rows = session.execute(
                select(WeatherDayModel.obs_date)
                .join(WeatherSourceModel, WeatherDayModel.source_id == WeatherSourceModel.id)
                .where(
                    WeatherDayModel.grid_point_id == grid_point_id,
                    WeatherSourceModel.model == self.settings.model,
                    WeatherDayModel.obs_date >= start,
                    WeatherDayModel.obs_date <= end,
                )
            ).scalars().all()
            return set(rows)

    # ────────────────────────────────────────────────────────────────────
    # ВНУТРЕННЯЯ ЛОГИКА: сохранение сегмента
    # ────────────────────────────────────────────────────────────────────

    def _save_segment(
        self,
        grid_point_id: int,
        start_date: str,
        end_date: str,
        payload: dict,
        source_url: str,
    ) -> None:
        daily = payload.get("daily", {})
        daily_units = payload.get("daily_units", {})
        times = daily.get("time", [])
        speeds = daily.get(WIND_SPEED_VAR, [])
        directions = daily.get(WIND_DIRECTION_VAR, [])

        ws_unit = daily_units.get(WIND_SPEED_VAR)
        wd_unit = daily_units.get(WIND_DIRECTION_VAR)
        variables_key = "wind_speed_direction"

        source_id = self._get_or_create_source_id(
            model=self.settings.model,
            ws_unit=ws_unit,
            wd_unit=wd_unit,
        )

        with session_scope() as session:
            grid_point = session.get(GridPointModel, grid_point_id)
            if grid_point is None:
                raise KeyError(f"Grid point not found: id={grid_point_id}")

            grid_point.resolved_lat = payload.get("latitude", grid_point.resolved_lat)
            grid_point.resolved_lon = payload.get("longitude", grid_point.resolved_lon)
            grid_point.elevation_m = payload.get("elevation", grid_point.elevation_m)

            for idx, raw_day in enumerate(times):
                obs_date = self._parse_date(raw_day)
                wind_speed_max = speeds[idx] if idx < len(speeds) else None
                wind_direction = directions[idx] if idx < len(directions) else None

                existing_day = session.execute(
                    select(WeatherDayModel).where(
                        WeatherDayModel.grid_point_id == grid_point_id,
                        WeatherDayModel.source_id == source_id,
                        WeatherDayModel.obs_date == obs_date,
                    )
                ).scalar_one_or_none()

                if existing_day is not None:
                    existing_day.wind_speed_max = wind_speed_max
                    existing_day.wind_direction = wind_direction
                else:
                    session.add(
                        WeatherDayModel(
                            grid_point_id=grid_point_id,
                            source_id=source_id,
                            obs_date=obs_date,
                            wind_speed_max=wind_speed_max,
                            wind_direction=wind_direction,
                        )
                    )

            # Только метаданные сегмента — без копии payload и без
            # копии значений ветра (они уже сохранены выше).
            segment = DownloadSegmentModel(
                grid_point_id=grid_point_id,
                source_id=source_id,
                start_date=self._parse_date(start_date),
                end_date=self._parse_date(end_date),
                variables_key=variables_key,
                timezone=self.settings.timezone,
                cell_selection=self.settings.cell_selection,
                source_url=source_url,
                downloaded_at=datetime.now(UTC),
            )
            session.add(segment)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                logger.debug(
                    f"Download segment already recorded: grid_point_id={grid_point_id}, "
                    f"range={start_date}..{end_date}"
                )

    # ────────────────────────────────────────────────────────────────────
    # ВНУТРЕННЯЯ ЛОГИКА: батчинг и диапазоны дат
    # ────────────────────────────────────────────────────────────────────

    def _build_download_tasks(
        self,
        grid_points: list[GridPoint],
        grid_point_ids: list[int],
        point_missing_ranges: dict[int, list[tuple[str, str]]],
    ) -> list[dict]:
        tasks: list[dict] = []
        for point, grid_point_id in zip(grid_points, grid_point_ids, strict=True):
            ranges = point_missing_ranges.get(grid_point_id)
            if not ranges:
                continue
            for range_start, range_end in ranges:
                tasks.append(
                    {
                        "point": point,
                        "grid_point_id": grid_point_id,
                        "start_date": range_start,
                        "end_date": range_end,
                    }
                )
        return tasks

    @staticmethod
    def _group_tasks_by_date_range(tasks: list[dict]) -> dict[tuple[str, str], list[dict]]:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for task in tasks:
            key = (task["start_date"], task["end_date"])
            grouped.setdefault(key, []).append(task)
        return grouped

    @staticmethod
    def _batched(items: list[dict], size: int) -> list[list[dict]]:
        return [items[i:i + size] for i in range(0, len(items), size)]

    def _normalize_requested_range(self, start_date: str, end_date: str) -> tuple[str, str]:
        requested_start = self._parse_date(start_date)
        requested_end = self._parse_date(end_date)

        if requested_start > requested_end:
            raise ValueError(f"start_date={start_date} is after end_date={end_date}")

        min_allowed = self._parse_date(self.settings.archive_min_date)
        current_utc_date = datetime.now(UTC).date()
        max_allowed = current_utc_date - timedelta(days=self.settings.archive_lag_days)

        normalized_start = max(requested_start, min_allowed)
        normalized_end = min(requested_end, max_allowed)

        if normalized_start > normalized_end:
            raise ValueError(
                "Requested range is outside historical archive coverage. "
                f"Allowed: {min_allowed.isoformat()}..{max_allowed.isoformat()}"
            )

        return normalized_start.isoformat(), normalized_end.isoformat()

    @staticmethod
    def _parse_date(value: str) -> date:
        return date.fromisoformat(value)

    def _date_set(self, start_date: str, end_date: str) -> set[date]:
        return set(self._date_iter(start_date, end_date))

    def _date_iter(self, start_date: str, end_date: str):
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _dates_to_ranges(dates: list[date]) -> list[tuple[str, str]]:
        if not dates:
            return []
        ranges: list[tuple[str, str]] = []
        range_start = dates[0]
        prev = dates[0]
        for current in dates[1:]:
            if (current - prev).days == 1:
                prev = current
                continue
            ranges.append((range_start.isoformat(), prev.isoformat()))
            range_start = current
            prev = current
        ranges.append((range_start.isoformat(), prev.isoformat()))
        return ranges
