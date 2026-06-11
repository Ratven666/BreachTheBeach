from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import Point

from .GeoJsonGridBuilder import GeoJsonGridBuilder
from .OpenMeteoArchiveClient import OpenMeteoArchiveClient
from .WeatherCache import WeatherCache
from .WeatherDownloadConfig import WeatherDownloadConfig
from .models import GridPoint, WeatherRequest


class WeatherHistoryService:
    def __init__(self, config: WeatherDownloadConfig | None = None) -> None:
        self.config = config or WeatherDownloadConfig()
        self.grid_builder = GeoJsonGridBuilder(
            grid_step=self.config.grid_step,
            grid_center_offset=self.config.grid_center_offset,
            cover_points_with_cells=self.config.cover_points_with_cells,
            extra_border_cells=self.config.extra_border_cells,
        )
        self.cache = WeatherCache(self.config.cache_dir)
        self.client = OpenMeteoArchiveClient(user_agent=self.config.user_agent)

    def build_request(
        self,
        geojson_path: str | Path,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...] | None = None,
    ) -> WeatherRequest:
        normalized_start, normalized_end = self._normalize_requested_range(
            start_date=start_date,
            end_date=end_date,
        )

        return WeatherRequest(
            geojson_path=Path(geojson_path),
            start_date=normalized_start,
            end_date=normalized_end,
            daily_variables=daily_variables or self.config.daily_variables,
        )

    def download_from_geojson(
        self,
        geojson_path: str | Path,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...] | None = None,
        output_geojson_path: str | Path | None = None,
    ) -> dict:
        request = self.build_request(
            geojson_path=geojson_path,
            start_date=start_date,
            end_date=end_date,
            daily_variables=daily_variables,
        )

        output_geojson = Path(output_geojson_path or self.config.output_geojson_path)
        output_geojson.parent.mkdir(parents=True, exist_ok=True)

        source_bbox, weather_bbox, grid_points = self.grid_builder.build_grid(request.geojson_path)
        logger.info(f"Source bbox: {source_bbox}")
        logger.info(f"Weather bbox: {weather_bbox}")
        logger.info(f"Grid points generated: {len(grid_points)}")
        logger.info(
            f"Coverage mode: cells_cover_points={self.config.cover_points_with_cells}, "
            f"extra_border_cells={self.config.extra_border_cells}"
        )
        logger.info(
            f"Effective weather period: {request.start_date}..{request.end_date}"
        )

        point_missing_ranges: dict[tuple[float, float, int, int], list[tuple[str, str]]] = {}
        fully_cached_points = 0
        partially_missing_points = 0

        for point in grid_points:
            missing_ranges = self._get_missing_ranges_for_point(
                point=point,
                start_date=request.start_date,
                end_date=request.end_date,
                daily_variables=request.daily_variables,
            )

            if not missing_ranges:
                fully_cached_points += 1
                logger.debug(
                    f"Point lat={point.lat:.3f}, lon={point.lon:.3f}: full cache hit"
                )
            else:
                partially_missing_points += 1
                point_missing_ranges[self._point_key(point)] = missing_ranges
                logger.info(
                    f"Point lat={point.lat:.3f}, lon={point.lon:.3f}, "
                    f"ring_y={point.ring_y}, ring_x={point.ring_x}, "
                    f"missing_ranges={missing_ranges}"
                )

        logger.info(
            f"Fully cached points: {fully_cached_points}, "
            f"points needing download: {partially_missing_points}"
        )

        all_download_tasks = self._build_download_tasks(grid_points, point_missing_ranges)
        logger.info(f"Download tasks: {len(all_download_tasks)}")

        grouped_tasks = self._group_tasks_by_date_range(all_download_tasks)

        for (batch_start, batch_end), points_for_range in grouped_tasks:
            for batch in self._batched(points_for_range, self.config.batch_size):
                payload, source_url = self.client.fetch(
                    points=batch,
                    start_date=batch_start,
                    end_date=batch_end,
                    daily_variables=request.daily_variables,
                    model=self.config.model,
                    timezone=self.config.timezone,
                    cell_selection=self.config.cell_selection,
                )

                records = payload if isinstance(payload, list) else [payload]

                if len(records) != len(batch):
                    raise ValueError(
                        f"Response size mismatch: received {len(records)} records "
                        f"for {len(batch)} requested points"
                    )

                for point, record in zip(batch, records, strict=True):
                    self.cache.save_segment(
                        point=point,
                        model=self.config.model,
                        start_date=batch_start,
                        end_date=batch_end,
                        daily_variables=request.daily_variables,
                        payload=record,
                        source_url=source_url,
                        timezone=self.config.timezone,
                        cell_selection=self.config.cell_selection,
                    )
                    logger.info(
                        f"Saved weather cache for point "
                        f"lat={point.lat:.3f}, lon={point.lon:.3f}, "
                        f"ring_y={point.ring_y}, ring_x={point.ring_x}, "
                        f"range={batch_start}..{batch_end}"
                    )

                if self.config.request_pause_seconds > 0:
                    time.sleep(self.config.request_pause_seconds)

        missing_after_download = [
            point
            for point in grid_points
            if self._get_missing_ranges_for_point(
                point=point,
                start_date=request.start_date,
                end_date=request.end_date,
                daily_variables=request.daily_variables,
            )
        ]
        if missing_after_download:
            raise RuntimeError(
                f"Dataset is incomplete. Missing points after download: {len(missing_after_download)}"
            )

        gdf = self._build_output_layer(
            points=grid_points,
            start_date=request.start_date,
            end_date=request.end_date,
            daily_variables=request.daily_variables,
        )

        gdf.to_file(output_geojson, driver="GeoJSON")
        logger.success(f"Weather GeoJSON saved: {output_geojson}")

        return {
            "source_bbox": source_bbox,
            "weather_bbox": weather_bbox,
            "grid_points_count": len(grid_points),
            "download_tasks_count": len(all_download_tasks),
            "effective_start_date": request.start_date,
            "effective_end_date": request.end_date,
            "output_geojson_path": str(output_geojson),
        }

    def _build_output_layer(
        self,
        points: list[GridPoint],
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
    ) -> gpd.GeoDataFrame:
        rows: list[dict] = []

        for index, point in enumerate(points, start=1):
            payload = self._merge_cached_payloads_for_period(
                point=point,
                start_date=start_date,
                end_date=end_date,
                daily_variables=daily_variables,
            )

            daily = payload.get("daily", {})
            daily_units = payload.get("daily_units", {})

            rows.append(
                {
                    "point_id": index,
                    "req_lat": point.lat,
                    "req_lon": point.lon,
                    "lat": payload.get("latitude"),
                    "lon": payload.get("longitude"),
                    "elev_m": payload.get("elevation"),
                    "tz": payload.get("timezone"),
                    "tz_abbr": payload.get("timezone_abbreviation"),
                    "start_date": start_date,
                    "end_date": end_date,
                    "dates": json.dumps(daily.get("time", []), ensure_ascii=False, separators=(",", ":")),
                    "wind_speed": json.dumps(
                        daily.get("wind_speed_10m_max", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "wind_dir": json.dumps(
                        daily.get("wind_direction_10m_dominant", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "ws_unit": daily_units.get("wind_speed_10m_max"),
                    "wd_unit": daily_units.get("wind_direction_10m_dominant"),
                    "ring_y": point.ring_y,
                    "ring_x": point.ring_x,
                    "geometry": Point(payload.get("longitude"), payload.get("latitude")),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    def _normalize_requested_range(self, start_date: str, end_date: str) -> tuple[str, str]:
        requested_start = self._parse_date(start_date)
        requested_end = self._parse_date(end_date)

        if requested_start > requested_end:
            raise ValueError(
                f"Invalid date range: start_date={start_date} is after end_date={end_date}"
            )

        if self.config.archive_lag_days < 0:
            raise ValueError(
                f"archive_lag_days must be >= 0, got {self.config.archive_lag_days}"
            )

        min_allowed = self._parse_date(self.config.archive_min_date)
        current_utc_date = datetime.now(UTC).date()
        max_allowed = current_utc_date - timedelta(days=self.config.archive_lag_days)

        normalized_start = max(requested_start, min_allowed)
        normalized_end = min(requested_end, max_allowed)

        if (normalized_start, normalized_end) != (requested_start, requested_end):
            logger.warning(
                f"Requested weather range adjusted from "
                f"{requested_start.isoformat()}..{requested_end.isoformat()} to "
                f"{normalized_start.isoformat()}..{normalized_end.isoformat()} "
                f"(lag_days={self.config.archive_lag_days})"
            )

        if normalized_start > normalized_end:
            raise ValueError(
                "Requested range is outside historical archive coverage. "
                f"Allowed range: {min_allowed.isoformat()}..{max_allowed.isoformat()} "
                f"(current_utc_date={current_utc_date.isoformat()}, "
                f"lag_days={self.config.archive_lag_days}), "
                f"requested: {requested_start.isoformat()}..{requested_end.isoformat()}"
            )

        return normalized_start.isoformat(), normalized_end.isoformat()

    def _get_missing_ranges_for_point(
        self,
        point: GridPoint,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        requested_dates = self._date_set(start_date, end_date)
        cached_dates = self._collect_cached_dates_for_point(
            point=point,
            start_date=start_date,
            end_date=end_date,
            daily_variables=daily_variables,
        )
        missing_dates = sorted(requested_dates - cached_dates)
        return self._dates_to_ranges(missing_dates)

    def _collect_cached_dates_for_point(
        self,
        point: GridPoint,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
    ) -> set[date]:
        covered: set[date] = set()
        request_start = self._parse_date(start_date)
        request_end = self._parse_date(end_date)

        for meta in self.cache.iter_segments_metadata(
            point=point,
            model=self.config.model,
            daily_variables=daily_variables,
            timezone=self.config.timezone,
            cell_selection=self.config.cell_selection,
        ):
            seg_start = self._parse_date(meta["start_date"])
            seg_end = self._parse_date(meta["end_date"])

            if seg_end < request_start or seg_start > request_end:
                continue

            payload = self.cache.load_segment_by_key(meta["cache_key"])
            valid_days = self._extract_cached_dates_from_payload(
                payload=payload,
                request_start=request_start,
                request_end=request_end,
            )
            covered.update(valid_days)

        return covered

    def _extract_cached_dates_from_payload(
        self,
        payload: dict,
        request_start: date,
        request_end: date,
    ) -> set[date]:
        daily = payload.get("daily", {})
        times = daily.get("time", [])

        result: set[date] = set()
        for raw_day in times:
            day = self._parse_date(raw_day)
            if request_start <= day <= request_end:
                result.add(day)

        return result

    def _merge_cached_payloads_for_period(
        self,
        point: GridPoint,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
    ) -> dict:
        request_start = self._parse_date(start_date)
        request_end = self._parse_date(end_date)
        expected_dates = self._date_list(start_date, end_date)

        per_day: dict[str, dict[str, object]] = {}
        base_payload: dict | None = None

        for meta in self.cache.iter_segments_metadata(
            point=point,
            model=self.config.model,
            daily_variables=daily_variables,
            timezone=self.config.timezone,
            cell_selection=self.config.cell_selection,
        ):
            seg_start = self._parse_date(meta["start_date"])
            seg_end = self._parse_date(meta["end_date"])

            if seg_end < request_start or seg_start > request_end:
                continue

            payload = self.cache.load_segment_by_key(meta["cache_key"])
            daily = payload.get("daily", {})
            times = daily.get("time", [])

            if not times:
                continue

            if base_payload is None:
                base_payload = payload

            for idx, raw_day in enumerate(times):
                day = self._parse_date(raw_day)
                if day < request_start or day > request_end:
                    continue

                day_key = day.isoformat()
                if day_key not in per_day:
                    per_day[day_key] = {}

                for variable in daily_variables:
                    values = daily.get(variable, [])
                    value = values[idx] if idx < len(values) else None
                    per_day[day_key][variable] = value

        missing_days = [
            day_key
            for day_key in expected_dates
            if day_key not in per_day
        ]
        if missing_days:
            raise RuntimeError(
                f"Missing cached dates for point lat={point.lat}, lon={point.lon}: "
                f"{missing_days[:10]}{'...' if len(missing_days) > 10 else ''}"
            )

        base = base_payload or {}
        base_units = base.get("daily_units", {})

        merged_daily = {"time": expected_dates}
        for variable in daily_variables:
            merged_daily[variable] = [per_day[day_key].get(variable) for day_key in expected_dates]

        return {
            "latitude": base.get("latitude", point.lat),
            "longitude": base.get("longitude", point.lon),
            "elevation": base.get("elevation"),
            "timezone": base.get("timezone", self.config.timezone),
            "timezone_abbreviation": base.get("timezone_abbreviation", self.config.timezone),
            "daily_units": {
                "time": "iso8601",
                **{variable: base_units.get(variable) for variable in daily_variables},
            },
            "daily": merged_daily,
        }

    def _build_download_tasks(
        self,
        grid_points: list[GridPoint],
        point_missing_ranges: dict[tuple[float, float, int, int], list[tuple[str, str]]],
    ) -> list[dict]:
        points_by_key = {self._point_key(point): point for point in grid_points}
        tasks: list[dict] = []

        for point_key, ranges in point_missing_ranges.items():
            point = points_by_key[point_key]
            for range_start, range_end in ranges:
                tasks.append(
                    {
                        "point": point,
                        "start_date": range_start,
                        "end_date": range_end,
                    }
                )

        tasks.sort(
            key=lambda item: (
                item["start_date"],
                item["end_date"],
                item["point"].ring_y,
                item["point"].ring_x,
            )
        )
        return tasks

    def _group_tasks_by_date_range(
        self,
        tasks: list[dict],
    ) -> list[tuple[tuple[str, str], list[GridPoint]]]:
        grouped: dict[tuple[str, str], list[GridPoint]] = {}

        for task in tasks:
            key = (task["start_date"], task["end_date"])
            grouped.setdefault(key, []).append(task["point"])

        return sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))

    @staticmethod
    def _point_key(point: GridPoint) -> tuple[float, float, int, int]:
        return (point.lat, point.lon, point.ring_y, point.ring_x)

    @staticmethod
    def _parse_date(value: str) -> date:
        return datetime.strptime(value, "%Y-%m-%d").date()

    def _date_set(self, start_date: str, end_date: str) -> set[date]:
        return set(self._date_iter(start_date, end_date))

    def _date_list(self, start_date: str, end_date: str) -> list[str]:
        return [d.isoformat() for d in self._date_iter(start_date, end_date)]

    def _date_iter(self, start_date: str, end_date: str):
        current = self._parse_date(start_date)
        end = self._parse_date(end_date)
        while current <= end:
            yield current
            current += timedelta(days=1)

    def _dates_to_ranges(self, dates: list[date]) -> list[tuple[str, str]]:
        if not dates:
            return []

        ranges: list[tuple[str, str]] = []
        range_start = dates[0]
        prev = dates[0]

        for current in dates[1:]:
            if current == prev + timedelta(days=1):
                prev = current
                continue

            ranges.append((range_start.isoformat(), prev.isoformat()))
            range_start = current
            prev = current

        ranges.append((range_start.isoformat(), prev.isoformat()))
        return ranges

    @staticmethod
    def _batched(items: list[GridPoint], size: int) -> list[list[GridPoint]]:
        return [items[index:index + size] for index in range(0, len(items), size)]
