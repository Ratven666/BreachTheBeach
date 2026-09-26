"""Шаг 8 пайплайна — расчёт волнового воздействия на берег."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from loguru import logger

from secret.OPEN_TOPOGRAPHY_API import OP_TOP_KEY
from src.base.BBox import BBox
from src.bathymetry import (
    BathymetryCache,
    BathymetryLoaderFactory,
    BathymetryService,
)
from src.bathymetry.domain.models import GeoLine, GeoPoint
from src.waves.services.wave_climate_batch import WaveClimateBatchProcessor

# ── Корень рабочей директории ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE         = PROJECT_ROOT / "nvrsk_calc"

# ── Входные файлы ─────────────────────────────────────────────────────────────
POINTS_PATH     = BASE / "nvrsk_equal_radius_1000m_points_with_normals.geojson"
FETCH_CSV       = BASE / "fetch" / "fetch_by_point.csv"
WEATHER_GEOJSON = BASE / "points_with_weather.geojson"

# ── Батиметрия: точно та же директория что в шаге 7 ──────────────────────────
BATHY_CASE_DIR  = BASE / "bathymetry_from_bbox" / "nvrsk_merged_dataset"
BATHY_CACHE_DIR = BATHY_CASE_DIR / "cache"
BATHY_RAW_DIR   = BATHY_CASE_DIR / "raw"

# GeoJSON который шаг 7 использовал для bbox — берём тот же
BATHY_SOURCE_GEOJSON = BASE / "merged_dataset.geojson"

# ── Выходные файлы ────────────────────────────────────────────────────────────
DAILY_OUT   = BASE / "wave_climate_daily.geojson"
SUMMARY_OUT = BASE / "wave_climate_summary.geojson"
WEATHER_CSV = BASE / "weather_expanded.csv"

# ── Параметры профиля ─────────────────────────────────────────────────────────
BATHY_RADIUS_M = 20_000.0
BATHY_N_STEPS  = 200

_GRID_EPS = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Геодезика
# ─────────────────────────────────────────────────────────────────────────────

def _endpoint(
    lat0: float, lon0: float, azimuth_deg: float, dist_m: float
) -> tuple[float, float]:
    R  = 6_371_000.0
    az = math.radians(azimuth_deg)
    φ0 = math.radians(lat0)
    λ0 = math.radians(lon0)
    d  = dist_m / R
    φ1 = math.asin(
        math.sin(φ0) * math.cos(d) + math.cos(φ0) * math.sin(d) * math.cos(az)
    )
    λ1 = λ0 + math.atan2(
        math.sin(az) * math.sin(d) * math.cos(φ0),
        math.cos(d) - math.sin(φ0) * math.sin(φ1),
    )
    return math.degrees(φ1), math.degrees(λ1)


def _grid_bounds(grid) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) из объекта грида."""
    if hasattr(grid, "bbox"):
        b = grid.bbox
        return float(b.west), float(b.east), float(b.south), float(b.north)
    if hasattr(grid, "bounds"):
        b = grid.bounds
        return float(b[0]), float(b[2]), float(b[1]), float(b[3])
    for lon_attr, lat_attr in (("lon", "lat"), ("lons", "lats")):
        if hasattr(grid, lon_attr) and hasattr(grid, lat_attr):
            lons = np.asarray(getattr(grid, lon_attr)).ravel()
            lats = np.asarray(getattr(grid, lat_attr)).ravel()
            return float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())
    raise AttributeError(
        f"Не удалось извлечь bounds из {type(grid).__name__}. "
        f"Доступные атрибуты: {[a for a in dir(grid) if not a.startswith('_')]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Адаптер профиля
# ─────────────────────────────────────────────────────────────────────────────

class _ProfileAdapter:
    __slots__ = ("_profile",)

    def __init__(self, profile: object) -> None:
        self._profile = profile

    @property
    def depths_m(self) -> np.ndarray:
        return np.asarray(self._profile.depths, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Per-point сервис батиметрии
# ─────────────────────────────────────────────────────────────────────────────

class _PointBathyService:
    def __init__(
        self,
        bathy_svc: BathymetryService,
        origin_lon: float,
        origin_lat: float,
        radius_m: float = BATHY_RADIUS_M,
        n_points: int   = BATHY_N_STEPS,
        grid_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self._svc      = bathy_svc
        self._radius_m = radius_m
        self._n_points = n_points
        self._cache: dict[int, _ProfileAdapter] = {}

        if grid_bounds is not None:
            lon_min, lon_max, lat_min, lat_max = grid_bounds
            self._lon = max(lon_min + _GRID_EPS, min(lon_max - _GRID_EPS, origin_lon))
            self._lat = max(lat_min + _GRID_EPS, min(lat_max - _GRID_EPS, origin_lat))
            if self._lon != origin_lon or self._lat != origin_lat:
                logger.debug(
                    f"Клип: ({origin_lon:.6f}, {origin_lat:.6f}) "
                    f"→ ({self._lon:.6f}, {self._lat:.6f})"
                )
        else:
            self._lon = origin_lon
            self._lat = origin_lat

    def get_profile(self, direction: int) -> _ProfileAdapter:
        direction = int(direction) % 360
        if direction in self._cache:
            return self._cache[direction]
        end_lat, end_lon = _endpoint(
            self._lat, self._lon, float(direction), self._radius_m
        )
        line = GeoLine(
            start=GeoPoint(lat=self._lat, lon=self._lon),
            end=GeoPoint(lat=end_lat,     lon=end_lon),
        )
        profile = self._svc.build_profile(line, n_points=self._n_points)
        self._cache[direction] = _ProfileAdapter(profile)
        return self._cache[direction]


# ─────────────────────────────────────────────────────────────────────────────
# Патченный батч-процессор
# ─────────────────────────────────────────────────────────────────────────────

class _PatchedBatchProcessor(WaveClimateBatchProcessor):
    def __init__(
        self,
        bathy_svc: Optional[BathymetryService],
        grid_bounds: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bathy_svc   = bathy_svc
        self._grid_bounds = grid_bounds

    def run(
        self,
        points_gdf,
        fetch_df,
        weather_csv_df,
        normal_field: str = "normal_azimuth_deg",
        bathymetry_service=None,
    ):
        import pandas as pd
        from src.waves.services.wave_climate_service import WaveClimateService as _WCS
        from src.waves.services.wave_climate_batch import (
            _normalize_point_id,
            _build_fetch_lookup_all,
        )
        from loguru import logger as _log

        if "point_id" not in points_gdf.columns:
            points_gdf = points_gdf.copy()
            points_gdf["point_id"] = range(len(points_gdf))

        points_gdf = points_gdf.copy()
        points_gdf["_pid_int"] = points_gdf["point_id"].apply(_normalize_point_id)

        points_wgs84 = points_gdf.to_crs("EPSG:4326")

        weather = weather_csv_df.copy()
        weather["_pid_int"] = weather["point_id"].apply(_normalize_point_id)
        weather["date"]     = pd.to_datetime(weather["date"])
        weather_by_pid: dict[int, pd.DataFrame] = {
            pid: grp.drop(columns=["_pid_int"])
            for pid, grp in weather.groupby("_pid_int")
        }
        _log.info(f"Метео сгруппировано для {len(weather_by_pid)} точек")

        fetch_lookup = _build_fetch_lookup_all(fetch_df.copy())

        daily_features:   list[dict] = []
        summary_features: list[dict] = []
        total  = len(points_gdf)
        errors = 0

        for i, ((_, pt_row), (_, pt_wgs)) in enumerate(
            zip(points_gdf.iterrows(), points_wgs84.iterrows())
        ):
            pid_int = int(pt_row["_pid_int"])
            geom    = pt_row.geometry
            normal  = float(pt_row[normal_field])
            orig_id = pt_row["point_id"]

            lon_wgs = float(pt_wgs.geometry.x)
            lat_wgs = float(pt_wgs.geometry.y)

            wdf = weather_by_pid.get(pid_int)
            if wdf is None or wdf.empty:
                _log.warning(f"[{i}/{total}] pid={pid_int}: нет метеоданных, пропускаем")
                errors += 1
                continue

            fetch_pid = pid_int + self.fetch_id_offset
            trace_df  = fetch_lookup.get(fetch_pid)
            if trace_df is None:
                trace_df = fetch_lookup.get(pid_int)
            if trace_df is None:
                _log.warning(
                    f"[{i}/{total}] pid={pid_int} (fetch_pid={fetch_pid}): "
                    "нет fetch, пропускаем"
                )
                errors += 1
                continue

            point_bathy = (
                _PointBathyService(
                    bathy_svc=self._bathy_svc,
                    origin_lon=lon_wgs,
                    origin_lat=lat_wgs,
                    radius_m=self.bathy_radius_m,
                    n_points=self.bathy_n_steps,
                    grid_bounds=self._grid_bounds,
                )
                if self._bathy_svc is not None
                else None
            )

            try:
                svc = _WCS(
                    trace_df=trace_df,
                    wind_ts_df=wdf,
                    shore_normal_deg=normal,
                    bathymetry_service=point_bathy,
                    origin_lon=lon_wgs,
                    origin_lat=lat_wgs,
                    bathy_radius_m=self.bathy_radius_m,
                    bathy_n_steps=self.bathy_n_steps,
                    overwater_factor=self.overwater_factor,
                    breaking_coeff=self.breaking_coeff,
                    rho_water=self.rho_water,
                    g=self.g,
                    default_h_deep_m=self.default_h_deep_m,
                    default_h_point_m=self.default_h_point_m,
                )
                daily = svc.calculate_daily()
            except Exception as exc:
                _log.error(f"[{i}/{total}] pid={pid_int}: ошибка расчёта: {exc}")
                errors += 1
                continue

            if daily.empty:
                _log.warning(f"[{i}/{total}] pid={pid_int}: пустой дневной ряд")
                continue

            for _, row in daily.iterrows():
                props = row.to_dict()
                if hasattr(props.get("date"), "isoformat"):
                    props["date"] = props["date"].isoformat()
                props["point_id"]         = orig_id
                props["point_id_int"]     = pid_int
                props["shore_normal_deg"] = round(normal, 2)
                props["geometry"]         = geom
                daily_features.append(props)

            stats = svc.cwef_stats(daily)
            stats["point_id"]     = orig_id
            stats["point_id_int"] = pid_int
            stats["geometry"]     = geom
            skip = {"geometry", "_pid_int"}
            for col in points_gdf.columns:
                if col not in skip and col not in stats:
                    stats[col] = pt_row[col]
            summary_features.append(stats)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                _log.info(f"  Обработано {i + 1}/{total}, ошибок: {errors}")

        _log.info(
            f"Расчёт завершён: {len(summary_features)} точек успешно, {errors} ошибок"
        )

        daily_gdf = (
            gpd.GeoDataFrame(daily_features, geometry="geometry", crs=points_gdf.crs)
            if daily_features else gpd.GeoDataFrame()
        )
        summary_gdf = (
            gpd.GeoDataFrame(summary_features, geometry="geometry", crs=points_gdf.crs)
            if summary_features else gpd.GeoDataFrame()
        )
        return daily_gdf, summary_gdf


# ─────────────────────────────────────────────────────────────────────────────
# Инициализация BathymetryService из кэша шага 7
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_from_geojson(path: Path) -> BBox:
    """Тот же алгоритм что в шаге 7 — без буфера, из merged_dataset.geojson."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return BBox(
        south=float(miny),
        west=float(minx),
        north=float(maxy),
        east=float(maxx),
    )


def _build_bathymetry_service(bbox: BBox) -> BathymetryService:
    """Переиспользует кэш шага 7 — сеть не нужна если кэш есть."""
    if not BATHY_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Кэш батиметрии не найден: {BATHY_CACHE_DIR}\n"
            "Сначала выполните шаг 7 (bathymetry_pipeline.py)."
        )

    BATHY_RAW_DIR.mkdir(parents=True, exist_ok=True)

    factory = BathymetryLoaderFactory(
        emodnet_output_dir=BATHY_RAW_DIR / "emodnet",
        emodnet_save_download=True,
        gebco_output_dir=BATHY_RAW_DIR / "gebco",
        gebco_save_download=True,
        gebco_api_key=OP_TOP_KEY,
    )
    loader = factory.create(bbox)
    logger.info(f"Источник батиметрии: {loader.source_name}")

    return BathymetryService(
        loader=loader,
        cache=BathymetryCache(BATHY_CACHE_DIR),
        n_profile_points=BATHY_N_STEPS,
        interp_method="linear",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    for path in (POINTS_PATH, FETCH_CSV, WEATHER_GEOJSON):
        if not path.exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {path}\n"
                "Убедитесь, что предыдущие шаги пайплайна выполнены."
            )

    # bbox из того же источника что и шаг 7
    logger.info(f"Читаем bbox из: {BATHY_SOURCE_GEOJSON}")
    bbox = _bbox_from_geojson(BATHY_SOURCE_GEOJSON)
    logger.info(
        f"BBox: S={bbox.south:.5f} W={bbox.west:.5f} "
        f"N={bbox.north:.5f} E={bbox.east:.5f}"
    )

    bathy_svc = _build_bathymetry_service(bbox)

    logger.info("Загрузка батиметрического грида (кэш шага 7)...")
    grid = bathy_svc.fetch(bbox)
    logger.info(
        f"Грид готов: shape={grid.shape}, "
        f"глубины [{grid.min_depth:.1f}, {grid.max_depth:.1f}] м"
    )

    try:
        gb = _grid_bounds(grid)
        logger.info(
            f"Границы грида: lon=[{gb[0]:.5f}, {gb[1]:.5f}], "
            f"lat=[{gb[2]:.5f}, {gb[3]:.5f}]"
        )
    except AttributeError as e:
        logger.warning(f"Не удалось извлечь bounds грида: {e}. Клипирование отключено.")
        gb = None

    processor = _PatchedBatchProcessor(
        bathy_svc=bathy_svc,
        grid_bounds=gb,
        bathy_radius_m=BATHY_RADIUS_M,
        bathy_n_steps=BATHY_N_STEPS,
    )

    logger.info("Запуск волнового пайплайна (шаг 8)...")
    daily_path, summary_path = processor.export(
        points_path=POINTS_PATH,
        fetch_csv_path=FETCH_CSV,
        weather_geojson_path=WEATHER_GEOJSON,
        weather_csv_path=WEATHER_CSV,
        daily_output_path=DAILY_OUT,
        summary_output_path=SUMMARY_OUT,
    )

    logger.success(f"Дневной ряд:   {daily_path}")
    logger.success(f"Сводная карта: {summary_path}")


if __name__ == "__main__":
    main()