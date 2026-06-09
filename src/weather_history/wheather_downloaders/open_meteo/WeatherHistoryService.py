from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import Point

from .WeatherCache import WeatherCache
from .WeatherDownloadConfig import WeatherDownloadConfig
from .GeoJsonGridBuilder import GeoJsonGridBuilder
from .models import GridPoint, WeatherRequest
from .OpenMeteoArchiveClient import OpenMeteoArchiveClient


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
        return WeatherRequest(
            geojson_path=Path(geojson_path),
            start_date=start_date,
            end_date=end_date,
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

        missing_points = [
            point
            for point in grid_points
            if not self.cache.has_segment(
                point=point,
                model=self.config.model,
                start_date=request.start_date,
                end_date=request.end_date,
                daily_variables=request.daily_variables,
            )
        ]

        logger.info(
            f"Cached points: {len(grid_points) - len(missing_points)}, "
            f"missing points: {len(missing_points)}"
        )

        for batch in self._batched(missing_points, self.config.batch_size):
            payload, source_url = self.client.fetch(
                points=batch,
                start_date=request.start_date,
                end_date=request.end_date,
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
                    start_date=request.start_date,
                    end_date=request.end_date,
                    daily_variables=request.daily_variables,
                    payload=record,
                    source_url=source_url,
                    timezone=self.config.timezone,
                    cell_selection=self.config.cell_selection,
                )
                logger.info(
                    f"Saved weather cache for point "
                    f"lat={point.lat:.3f}, lon={point.lon:.3f}, "
                    f"ring_y={point.ring_y}, ring_x={point.ring_x}"
                )

            if self.config.request_pause_seconds > 0:
                time.sleep(self.config.request_pause_seconds)

        missing_after_download = [
            point
            for point in grid_points
            if not self.cache.has_segment(
                point=point,
                model=self.config.model,
                start_date=request.start_date,
                end_date=request.end_date,
                daily_variables=request.daily_variables,
            )
        ]
        if missing_after_download:
            raise RuntimeError(f"Dataset is incomplete. Missing points: {len(missing_after_download)}")

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
            payload = self.cache.load_segment(
                point=point,
                model=self.config.model,
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

    @staticmethod
    def _batched(items: list[GridPoint], size: int) -> list[list[GridPoint]]:
        return [items[index:index + size] for index in range(0, len(items), size)]
