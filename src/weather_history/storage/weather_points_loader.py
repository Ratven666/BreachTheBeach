"""
src/weather_history/storage/weather_points_loader.py

Класс, принимающий geojson-файл с готовыми точками сетки (например,
era5_coastal_grid_nodes_025deg.geojson — точки-центры ячеек ERA5,
пересекаемых береговой линией) и организующий их загрузку в БД
через WeatherRepository. Визуализация прогресса — через tqdm.

Ожидаемый формат входного файла: FeatureCollection точек с properties,
содержащими как минимум координаты узла (era5_lon/era5_lat или lon/lat)
и, опционально, cell_id.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from loguru import logger
from tqdm import tqdm

from src.weather_history.storage.weather_repository import (
    WeatherDownloadSettings,
    WeatherRepository,
)
from src.weather_history.wheather_downloaders.open_meteo.models import GridPoint

LON_PROPERTY_CANDIDATES = ("era5_lon", "lon", "longitude", "req_lon")
LAT_PROPERTY_CANDIDATES = ("era5_lat", "lat", "latitude", "req_lat")
CELL_ID_PROPERTY_CANDIDATES = ("cell_id",)


class WeatherPointsLoader:
    """
    Определяет логику вызова загрузчиков БД для готового набора точек:
    - читает geojson с точками;
    - извлекает координаты и cell_id из каждой точки (из properties или геометрии);
    - разбивает точки на батчи и скачивает недостающие данные через WeatherRepository;
    - показывает прогресс скачивания через tqdm.
    """

    def __init__(
        self,
        points_path: str | Path,
        settings: WeatherDownloadSettings | None = None,
    ) -> None:
        self.points_path = Path(points_path)
        if not self.points_path.exists():
            raise FileNotFoundError(f"Points file not found: {self.points_path}")

        self.settings = settings or WeatherDownloadSettings()
        self.repository = WeatherRepository(settings=self.settings)

    # ────────────────────────────────────────────────────────────────────
    # ПУБЛИЧНЫЙ API
    # ────────────────────────────────────────────────────────────────────

    def load(self, start_date: str, end_date: str) -> dict:
        """
        Загружает данные о максимальной скорости ветра в БД для всех точек
        из файла, докачивая только недостающие диапазоны дат.
        Прогресс отображается через tqdm.
        """
        grid_points, cell_ids = self._read_points()
        logger.info(f"Loaded {len(grid_points)} points from {self.points_path}")

        progress_bar = tqdm(
            total=len(grid_points),
            desc="Проверка точек / скачивание погоды",
            unit="точка",
        )

        # Первый проход: узнаём, сколько задач (диапазонов на скачивание)
        # реально предстоит выполнить, чтобы прогресс-бар был осмысленным.
        missing_ranges_per_point = self._collect_missing_ranges(
            grid_points=grid_points,
            start_date=start_date,
            end_date=end_date,
            progress_bar=progress_bar,
        )
        progress_bar.close()

        total_tasks = sum(len(ranges) for ranges in missing_ranges_per_point)
        if total_tasks == 0:
            logger.success("Все точки уже загружены в БД, скачивание не требуется.")
            return {
                "grid_points_count": len(grid_points),
                "downloaded_segments": 0,
                "start_date": start_date,
                "end_date": end_date,
            }

        download_bar = tqdm(
            total=total_tasks,
            desc="Скачивание сегментов Open-Meteo",
            unit="сегмент",
        )

        def on_progress(done: int, total: int) -> None:
            download_bar.n = done
            download_bar.refresh()

        result = self.repository.download_for_points(
            grid_points=grid_points,
            start_date=start_date,
            end_date=end_date,
            cell_ids=cell_ids,
            progress_callback=on_progress,
        )
        download_bar.close()

        logger.success(f"Загрузка завершена: {result}")
        return result

    # ────────────────────────────────────────────────────────────────────
    # ВНУТРЕННЯЯ ЛОГИКА
    # ────────────────────────────────────────────────────────────────────

    def _read_points(self) -> tuple[list[GridPoint], list[str | None]]:
        gdf = gpd.read_file(self.points_path)
        if gdf.empty:
            raise ValueError(f"Points file is empty: {self.points_path}")

        grid_points: list[GridPoint] = []
        cell_ids: list[str | None] = []

        for _, row in gdf.iterrows():
            lon, lat = self._extract_coordinates(row)
            cell_id = self._extract_first_present(row, CELL_ID_PROPERTY_CANDIDATES)

            grid_points.append(GridPoint(lat=lat, lon=lon, ring_y=0, ring_x=0))
            cell_ids.append(str(cell_id) if cell_id is not None else None)

        return grid_points, cell_ids

    @staticmethod
    def _extract_first_present(row, candidates: tuple[str, ...]):
        for name in candidates:
            if name in row.index and row[name] is not None:
                return row[name]
        return None

    def _extract_coordinates(self, row) -> tuple[float, float]:
        lon = self._extract_first_present(row, LON_PROPERTY_CANDIDATES)
        lat = self._extract_first_present(row, LAT_PROPERTY_CANDIDATES)

        if lon is not None and lat is not None:
            return float(lon), float(lat)

        geometry = row.geometry
        if geometry is None or geometry.geom_type != "Point":
            raise ValueError(
                "Cannot extract coordinates: no lon/lat properties and "
                f"geometry is not a Point (got {getattr(geometry, 'geom_type', None)})"
            )
        return float(geometry.x), float(geometry.y)

    def _collect_missing_ranges(
        self,
        grid_points: list[GridPoint],
        start_date: str,
        end_date: str,
        progress_bar: tqdm,
    ) -> list[list[tuple[str, str]]]:
        normalized_start, normalized_end = self.repository._normalize_requested_range(
            start_date, end_date
        )

        results: list[list[tuple[str, str]]] = []
        for point in grid_points:
            grid_point_id = self.repository._get_or_create_grid_point(point)
            missing = self.repository._get_missing_ranges(
                grid_point_id=grid_point_id,
                start_date=normalized_start,
                end_date=normalized_end,
            )
            results.append(missing)
            progress_bar.update(1)

        return results
