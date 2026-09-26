"""
ERA5WindDownloader — загрузчик суточных данных о ветре из ERA5 CDS API.

Место в проекте: src/weather/ERA5WindDownloader.py

Зависимости (добавить в pyproject.toml):
    cdsapi = ">=0.7"
    xarray = ">=2024.1"
    netCDF4 = ">=1.7"      # или h5netcdf
    numpy = ">=1.26"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from secret.COPERNICUS_CLIMAT_DATA_STORE_API import CCDS_URL, CCDS_KEY

if TYPE_CHECKING:
    import xarray as xr
    import geopandas as gpd

# ---------------------------------------------------------------------------
# ERA5 resolution in degrees (0.25° × 0.25° grid)
# ---------------------------------------------------------------------------
ERA5_GRID_DEG: float = 0.25

# Wind variables we request: u10, v10 (daily mean, max, min) + gust (max)
# CDS variable names for derived-era5-single-levels-daily-statistics dataset
_WIND_VARIABLES: list[str] = [
    "10m_u_component_of_wind",       # u-составляющая (восток), м/с
    "10m_v_component_of_wind",       # v-составляющая (север), м/с
    "10m_wind_gust_since_previous_post_processing",  # порывы ветра, м/с
]

# Aggregation statistics requested for each variable
# mean — средняя; max — максимальная; min — минимальная скорость за сутки
_DAILY_STATISTICS: list[str] = ["daily_mean", "daily_max", "daily_min"]

# ERA5 temporal coverage start (data available from 1940-01-01)
ERA5_START_DATE: date = date(1940, 1, 1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindBBox:
    """
    Географический bounding box для запроса ERA5.

    Хранит координаты в WGS-84 (EPSG:4326).
    area-параметр CDS: [north, west, south, east].
    """
    north: float
    west: float
    south: float
    east: float

    def __post_init__(self) -> None:
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) must be < north ({self.north})")
        if self.west >= self.east:
            raise ValueError(f"west ({self.west}) must be < east ({self.east})")
        if not (-90.0 <= self.south <= 90.0 and -90.0 <= self.north <= 90.0):
            raise ValueError("Latitude must be in [-90, 90]")
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError("Longitude must be in [-180, 180]")

    def expanded(self, pixels: int = 1, grid_deg: float = ERA5_GRID_DEG) -> "WindBBox":
        """
        Расширяет bbox на `pixels` пикселей ERA5 во все стороны.

        По умолчанию 1 пиксель = 0.25°.  Используется для того, чтобы
        при интерполяции данных на береговые точки не было «обрезания».

        Parameters
        ----------
        pixels : int
            Количество ERA5-пикселей расширения.
        grid_deg : float
            Шаг сетки ERA5 в градусах (0.25 по умолчанию).
        """
        if pixels < 0:
            raise ValueError("pixels must be >= 0")
        delta = pixels * grid_deg
        return WindBBox(
            north=min(90.0, self.north + delta),
            west=max(-180.0, self.west - delta),
            south=max(-90.0, self.south - delta),
            east=min(180.0, self.east + delta),
        )

    def to_cds_area(self) -> list[float]:
        """CDS API area: [north, west, south, east]."""
        return [self.north, self.west, self.south, self.east]

    def __str__(self) -> str:
        return (
            f"WindBBox(N={self.north:.4f}, W={self.west:.4f}, "
            f"S={self.south:.4f}, E={self.east:.4f})"
        )


@dataclass
class ERA5WindRequest:
    """
    Параметры запроса суточных данных о ветре ERA5.

    Attributes
    ----------
    bbox : WindBBox
        Исходный bbox точек (без расширения).
    date_start : date | None
        Начало периода.  None → ERA5_START_DATE (1940-01-01).
    date_end : date | None
        Конец периода.  None → вчерашняя дата (последние доступные данные).
    expand_pixels : int
        Расширение bbox на n пикселей в каждую сторону (по умолчанию 1).
    output_path : Path
        Путь для сохранения NetCDF-файла.
    daily_statistics : list[str]
        Список суточных агрегаций.  По умолчанию mean + max + min.
    variables : list[str]
        CDS-имена переменных.  По умолчанию u10, v10, gust.
    frequency : str
        Частота дискретизации исходных часовых данных: "1_hourly" | "3_hourly" | "6_hourly".
    """
    bbox: WindBBox
    date_start: date | None = None
    date_end: date | None = None
    expand_pixels: int = 1
    output_path: Path = field(default_factory=lambda: Path("output/era5_wind.nc"))
    daily_statistics: list[str] = field(default_factory=lambda: list(_DAILY_STATISTICS))
    variables: list[str] = field(default_factory=lambda: list(_WIND_VARIABLES))
    frequency: str = "1_hourly"

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        if self.expand_pixels < 0:
            raise ValueError("expand_pixels must be >= 0")
        if self.frequency not in {"1_hourly", "3_hourly", "6_hourly"}:
            raise ValueError(
                f"frequency must be '1_hourly', '3_hourly', or '6_hourly', "
                f"got: '{self.frequency}'"
            )

    @property
    def effective_date_start(self) -> date:
        return self.date_start if self.date_start is not None else ERA5_START_DATE

    @property
    def effective_date_end(self) -> date:
        if self.date_end is not None:
            return self.date_end
        # ERA5 обновляется с задержкой ~6 дней
        return date.today() - timedelta(days=7)

    @property
    def effective_bbox(self) -> WindBBox:
        """bbox с расширением на expand_pixels."""
        return self.bbox.expanded(pixels=self.expand_pixels)


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class ERA5WindDownloader:
    """
    Загружает суточные данные о ветре ERA5 через Copernicus CDS API.

    Использует датасет:
        ``derived-era5-single-levels-daily-statistics``

    Переменные (по умолчанию):
        - 10m_u_component_of_wind        (u-составляющая, м/с)
        - 10m_v_component_of_wind        (v-составляющая, м/с)
        - 10m_wind_gust_since_previous_post_processing  (порывы, м/с)

    Суточные агрегаты (по умолчанию): mean, max, min.

    Расчёт скорости и направления ветра из u/v выполняется в методе
    ``compute_wind_speed_direction`` на загруженном xarray.Dataset.

    Examples
    --------
    Базовый пример — загрузка данных для набора береговых точек::

        from pathlib import Path
        from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
        from src.weather.ERA5WindDownloader import ERA5WindDownloader, ERA5WindRequest

        # Загружаем набор точек
        point_set = CoastlineNormalPointSet.from_geojson("output/points_with_normals.geojson")

        # Создаём запрос (bbox вычисляется автоматически из точек)
        request = ERA5WindDownloader.request_from_point_set(
            point_set=point_set,
            output_path=Path("output/era5_wind.nc"),
            date_start=date(2000, 1, 1),
            date_end=date(2023, 12, 31),
            expand_pixels=1,
        )

        # Скачиваем
        downloader = ERA5WindDownloader()
        ds = downloader.download(request)
        print(ds)
    """

    DATASET_ID: str = "derived-era5-single-levels-daily-statistics"

    def __init__(self,
                 cds_url: str | None = CCDS_URL,
                 cds_key: str | None = CCDS_KEY,
                 ) -> None:
        """
        Parameters
        ----------
        cds_url : str | None
            URL CDS API.  None → берётся из ~/.cdsapirc.
        cds_key : str | None
            API-ключ.  None → берётся из ~/.cdsapirc.
        """
        self._url = cds_url
        self._key = cds_key
        self._log = logger.bind(cls=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bbox_from_point_set(
        point_set: "CoastlineNormalPointSet",  # type: ignore[name-defined]
    ) -> WindBBox:
        """
        Вычисляет WindBBox из набора береговых точек.

        Точки могут быть в любой CRS — метод перепроецирует их в EPSG:4326
        перед вычислением bbox.
        """
        gdf = point_set.gdf
        if gdf.crs is None:
            raise ValueError("CoastlineNormalPointSet has no CRS")

        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

        minx, miny, maxx, maxy = gdf.total_bounds  # (west, south, east, north)
        return WindBBox(north=maxy, west=minx, south=miny, east=maxx)

    @classmethod
    def request_from_point_set(
        cls,
        point_set: "CoastlineNormalPointSet",  # type: ignore[name-defined]
        output_path: str | Path = "output/era5_wind.nc",
        date_start: date | None = None,
        date_end: date | None = None,
        expand_pixels: int = 1,
        daily_statistics: list[str] | None = None,
        variables: list[str] | None = None,
        frequency: str = "1_hourly",
    ) -> ERA5WindRequest:
        """
        Создаёт ERA5WindRequest из CoastlineNormalPointSet.

        Parameters
        ----------
        point_set :
            Набор береговых точек с нормалями.
        output_path :
            Путь для сохранения скачанного NetCDF.
        date_start :
            Начало периода.  None → весь доступный период ERA5 (с 1940-01-01).
        date_end :
            Конец периода.  None → последние доступные данные.
        expand_pixels :
            Расширение bbox в пикселях ERA5 (0.25°/пиксель) — для интерполяции.
        daily_statistics :
            Список статистик.  None → ['daily_mean', 'daily_max', 'daily_min'].
        variables :
            Список CDS-переменных.  None → u10 + v10 + gust.
        frequency :
            Частота дискретизации: '1_hourly' | '3_hourly' | '6_hourly'.
        """
        bbox = cls.bbox_from_point_set(point_set)
        return ERA5WindRequest(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            expand_pixels=expand_pixels,
            output_path=Path(output_path),
            daily_statistics=daily_statistics if daily_statistics is not None else list(_DAILY_STATISTICS),
            variables=variables if variables is not None else list(_WIND_VARIABLES),
            frequency=frequency,
        )

    @staticmethod
    def request_from_bbox(
        north: float,
        west: float,
        south: float,
        east: float,
        output_path: str | Path = "output/era5_wind.nc",
        date_start: date | None = None,
        date_end: date | None = None,
        expand_pixels: int = 1,
        daily_statistics: list[str] | None = None,
        variables: list[str] | None = None,
        frequency: str = "1_hourly",
    ) -> ERA5WindRequest:
        """
        Создаёт ERA5WindRequest из явно заданного bbox (WGS-84).
        """
        bbox = WindBBox(north=north, west=west, south=south, east=east)
        return ERA5WindRequest(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            expand_pixels=expand_pixels,
            output_path=Path(output_path),
            daily_statistics=daily_statistics if daily_statistics is not None else list(_DAILY_STATISTICS),
            variables=variables if variables is not None else list(_WIND_VARIABLES),
            frequency=frequency,
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def download(self, request: ERA5WindRequest) -> "xr.Dataset":
        """
        Скачивает суточные данные ERA5 о ветре и сохраняет NetCDF на диск.

        Возвращает открытый xarray.Dataset.

        Parameters
        ----------
        request : ERA5WindRequest
            Параметры запроса.

        Returns
        -------
        xr.Dataset
            Загруженный датасет ERA5.

        Raises
        ------
        ImportError
            Если не установлены cdsapi или xarray.
        ValueError
            Если date_start > date_end или нет данных по запросу.
        """
        import cdsapi  # noqa: PLC0415  (импорт внутри метода для ленивой загрузки)
        import xarray as xr

        start = request.effective_date_start
        end = request.effective_date_end
        eff_bbox = request.effective_bbox

        self._validate_dates(start, end)

        self._log.info(
            f"ERA5 download | dataset={self.DATASET_ID} | "
            f"period={start}..{end} | "
            f"bbox={eff_bbox} | "
            f"expand_pixels={request.expand_pixels} | "
            f"variables={request.variables} | "
            f"statistics={request.daily_statistics}"
        )

        cds_request = self._build_cds_request(request, start, end, eff_bbox)

        request.output_path.parent.mkdir(parents=True, exist_ok=True)

        client_kwargs: dict = {}
        if self._url:
            client_kwargs["url"] = self._url
        if self._key:
            client_kwargs["key"] = self._key

        client = cdsapi.Client(**client_kwargs)

        self._log.info(f"Sending CDS request, output → {request.output_path}")
        client.retrieve(self.DATASET_ID, cds_request, str(request.output_path))
        self._log.success(f"Downloaded: {request.output_path}")

        ds = xr.open_dataset(request.output_path)
        self._log.info(f"Dataset opened: {list(ds.data_vars)}")
        return ds

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_wind_speed_direction(ds: "xr.Dataset") -> "xr.Dataset":
        """
        Добавляет к датасету производные переменные скорости и направления ветра.

        Новые переменные:
        - wind_speed_{stat}   — скорость ветра, м/с (sqrt(u² + v²))
        - wind_direction_{stat} — метеорологическое направление, от куда дует, °

        Работает для каждой из имеющихся суточных статистик (mean / max / min).

        Parameters
        ----------
        ds : xr.Dataset
            Датасет, вернувшийся из download().
        """
        import numpy as np

        ds = ds.copy()

        for stat in ("daily_mean", "daily_max", "daily_min"):
            u_name = f"u10_{stat}"  # имена после скачивания из CDS
            v_name = f"v10_{stat}"

            # Попробуем альтернативные имена (CDS иногда использует короткие имена)
            u_candidates = [u_name, "u10", f"10m_u_component_of_wind_{stat}", "u10mean", "u10max", "u10min"]
            v_candidates = [v_name, "v10", f"10m_v_component_of_wind_{stat}", "v10mean", "v10max", "v10min"]

            u_var = next((n for n in u_candidates if n in ds), None)
            v_var = next((n for n in v_candidates if n in ds), None)

            if u_var is None or v_var is None:
                continue  # статистика не была запрошена

            u = ds[u_var]
            v = ds[v_var]

            # Скорость ветра
            speed = (u**2 + v**2) ** 0.5
            speed.attrs = {
                "long_name": f"10m wind speed ({stat})",
                "units": "m s**-1",
                "source_u": u_var,
                "source_v": v_var,
            }
            ds[f"wind_speed_{stat}"] = speed

            # Метеорологическое направление (откуда дует): 0° = С, 90° = В, ...
            # direction = (270 - atan2(v, u) * 180/π) % 360
            direction = (270.0 - np.degrees(np.arctan2(v.values, u.values))) % 360.0
            import xarray as xr
            dir_da = xr.DataArray(
                direction,
                coords=speed.coords,
                dims=speed.dims,
                attrs={
                    "long_name": f"10m wind direction ({stat}), meteorological convention",
                    "units": "degrees",
                    "convention": "direction wind is coming FROM, 0=N, 90=E, 180=S, 270=W",
                },
            )
            ds[f"wind_direction_{stat}"] = dir_da

        return ds

    @staticmethod
    def _validate_dates(start: date, end: date) -> None:
        today = date.today()
        if start > end:
            raise ValueError(
                f"date_start ({start}) must be <= date_end ({end})"
            )
        if start < ERA5_START_DATE:
            raise ValueError(
                f"ERA5 data is available from {ERA5_START_DATE}, "
                f"but date_start={start}"
            )
        if end >= today:
            logger.warning(
                f"date_end={end} is in the future or today; "
                "ERA5 has a ~6-day latency, recent data may be unavailable."
            )

    @staticmethod
    def _build_cds_request(
        request: ERA5WindRequest,
        start: date,
        end: date,
        bbox: WindBBox,
    ) -> dict:
        """
        Формирует словарь запроса для cdsapi.Client.retrieve().

        Запрос использует date-range нотацию CDS:
        ``"date": "YYYY-MM-DD/YYYY-MM-DD"``
        """
        return {
            "product_type": "reanalysis",
            "variable": request.variables,
            "statistic": request.daily_statistics,
            "frequency": request.frequency,
            "date": f"{start.isoformat()}/{end.isoformat()}",
            "area": bbox.to_cds_area(),   # [north, west, south, east]
            "data_format": "netcdf",
        }


# ---------------------------------------------------------------------------
# CLI entry point (python -m src.weather.ERA5WindDownloader)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Загрузка суточных данных ERA5 о ветре для заданного bbox "
            "или набора береговых точек."
        )
    )
    parser.add_argument("--north", type=float, required=True, help="Северная граница, °")
    parser.add_argument("--west", type=float, required=True, help="Западная граница, °")
    parser.add_argument("--south", type=float, required=True, help="Южная граница, °")
    parser.add_argument("--east", type=float, required=True, help="Восточная граница, °")
    parser.add_argument(
        "--start", type=str, default=None,
        help="Начало периода YYYY-MM-DD (по умолчанию 1940-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Конец периода YYYY-MM-DD (по умолчанию — последние доступные данные)",
    )
    parser.add_argument(
        "--expand", type=int, default=1,
        help="Расширение bbox в пикселях ERA5 (по умолчанию 1)",
    )
    parser.add_argument(
        "--output", type=str, default="output/era5_wind.nc",
        help="Путь для сохранения NetCDF (по умолчанию output/era5_wind.nc)",
    )
    parser.add_argument(
        "--frequency", type=str, default="1_hourly",
        choices=["1_hourly", "3_hourly", "6_hourly"],
        help="Частота дискретизации (по умолчанию 1_hourly)",
    )
    args = parser.parse_args()

    date_start = date.fromisoformat(args.start) if args.start else None
    date_end = date.fromisoformat(args.end) if args.end else None

    req = ERA5WindDownloader.request_from_bbox(
        north=args.north,
        west=args.west,
        south=args.south,
        east=args.east,
        output_path=args.output,
        date_start=date_start,
        date_end=date_end,
        expand_pixels=args.expand,
        frequency=args.frequency,
    )

    downloader = ERA5WindDownloader()
    try:
        ds = downloader.download(req)
        ds_with_derived = ERA5WindDownloader.compute_wind_speed_direction(ds)
        logger.success(f"Done. Variables: {list(ds_with_derived.data_vars)}")
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        sys.exit(1)
