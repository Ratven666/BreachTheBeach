"""Батч-процессор: расчёт волнового воздействия для всех точек трассы.

Читает входные файлы, обходит каждую точку, вызывает WaveClimateService,
собирает дневной ряд и сводную статистику, экспортирует в GeoJSON.

Соответствие ID точек
---------------------
• points GeoJSON    : point_id = int (CoastlineNormalPointSet, 0-based)
• fetch CSV         : point_id = int (из GDF точек, 0-based)
• weather GeoJSON   : point_id = int (WeatherLayerWrapper.assign_to_points)
• weather CSV (кэш) : point_id = int (нормализуется из GeoJSON)

fetch_id_offset = 0 (нумерация единая после исправления трассировщика).
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from src.waves.services.wave_climate_service import WaveClimateService


# ──────────────────────────────────────────────────────────────────
# вспомогательные функции
# ──────────────────────────────────────────────────────────────────

def _normalize_point_id(val) -> int:
    """Приводит любой вид point_id к целому числу.

    "point_00000" → 0 | "point_00001" → 1 | "1" → 1 | 1 → 1
    """
    s = str(val).strip()
    if s.startswith("point_"):
        return int(s[len("point_"):])
    return int(float(s))


def _parse_list_field(val) -> list | None:
    """Безопасно разбирает поле: numpy.ndarray / list / JSON-строка → list."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            try:
                return json.loads(val)
            except (ValueError, json.JSONDecodeError):
                return None
    return None


def _get_feat_field(feat, *names):
    """Возвращает первое непустое поле из pandas.Series.

    Не использует ``or`` на значениях — это вызывает ValueError на массивах.
    """
    for name in names:
        if name in feat.index:
            val = feat[name]
            if val is None:
                continue
            if not isinstance(val, (np.ndarray, list)) and pd.isna(val):
                continue
            return val
    return None


def _expand_weather_geojson(src: Path, dst: Path) -> None:
    """Разворачивает wide-GeoJSON метеоданных в длинный CSV.

    Поля в GeoJSON: dates, wind_speed, wind_dir (могут быть numpy-массивами).
    Результат CSV: point_id | date | wind_speed | wind_dir
    """
    if dst.exists() and dst.stat().st_size > 0:
        logger.info(f"Weather CSV уже существует, пропускаем: {dst}")
        return

    logger.info(f"Разворачиваем weather GeoJSON → CSV: {src}")
    gdf  = gpd.read_file(src)
    rows: list[dict] = []

    for _, feat in gdf.iterrows():
        raw_pid = _get_feat_field(feat, "point_id")
        if raw_pid is None:
            raw_pid = feat.name
        pid_int = _normalize_point_id(raw_pid)

        dates_raw = _get_feat_field(feat, "dates", "date")
        ws_raw    = _get_feat_field(feat, "wind_speed", "wind_speed_10m_max", "ws_kmh")
        wd_raw    = _get_feat_field(feat, "wind_dir", "wind_direction_10m_dominant", "direction")

        dates = _parse_list_field(dates_raw)
        ws    = _parse_list_field(ws_raw)
        wd    = _parse_list_field(wd_raw)

        if not dates or not ws or not wd:
            logger.warning(f"Точка {raw_pid}: нет метеоданных, пропускаем")
            continue

        if not (len(dates) == len(ws) == len(wd)):
            logger.warning(
                f"Точка {raw_pid}: длины списков не совпадают "
                f"(dates={len(dates)}, ws={len(ws)}, wd={len(wd)}), пропускаем"
            )
            continue

        for d, w, di in zip(dates, ws, wd):
            rows.append({
                "point_id":  pid_int,
                "date":      d,
                "wind_speed": w,   # единое имя с доменом WeatherPoint и WaveClimateService
                "wind_dir":   di,  # единое имя с доменом WeatherPoint и WaveClimateService
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("Weather CSV будет пустым — нет данных для записи!")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size < 10:
        dst.unlink()
    df.to_csv(dst, index=False)
    logger.success(f"Weather CSV сохранён ({len(df):,} строк): {dst}")


def _build_fetch_lookup_all(fetch_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Предварительно строит словарь {point_id_int → DataFrame(direction, fetch_m)}.

    Делается один раз перед циклом по точкам — избегаем повторного парсинга
    JSON-массивов на каждой итерации.
    Значения fetch_m == 0.1 — маркер суши, отфильтровываются.
    """
    result: dict[int, pd.DataFrame] = {}
    for _, row in fetch_df.iterrows():
        pid_int  = _normalize_point_id(row["point_id"])
        azimuths = _parse_list_field(row["azimuths_deg"])
        fetches  = _parse_list_field(row["fetch_lengths_m"])
        if not azimuths or not fetches or len(azimuths) != len(fetches):
            continue
        df = pd.DataFrame({"direction": azimuths, "fetch_m": fetches})
        df["direction"] = df["direction"].astype(float).round().astype(int) % 360
        df["fetch_m"]   = df["fetch_m"].astype(float)
        df = df[df["fetch_m"] > 0.1].copy()
        if not df.empty:
            result[pid_int] = df
    logger.info(f"Fetch lookup построен для {len(result)} точек")
    return result

def _resolve_point_bathy_service(bathymetry_service, lon, lat):
    """Если есть for_point() → создаёт per-point сервис. Иначе — as-is."""
    if bathymetry_service is None:
        return None
    factory_fn = getattr(bathymetry_service, "for_point", None)
    if callable(factory_fn):
        return factory_fn(lon, lat)   # ← ЧЕСТНЫЙ расчёт
    return bathymetry_service         # ← fallback совместимость


# ──────────────────────────────────────────────────────────────────
# батч-процессор
# ──────────────────────────────────────────────────────────────────

@dataclass
class WaveClimateBatchProcessor:
    """Батч-расчёт волнового воздействия по всем точкам трассы.

    Parameters
    ----------
    default_h_deep_m  : глубина в открытом море (без батиметрии), м
    default_h_point_m : глубина у берега (без батиметрии), м
    overwater_factor  : поправка скорости ветра над водой (рек. 1.1)
    breaking_coeff    : коэффициент обрушения γ_b (рек. 0.55)
    fetch_id_offset   : сдвиг между point_id точек и fetch CSV.
                        После унификации нумерации в трассировщике оба 0-based → offset=0.
    """
    overwater_factor:  float = 1.1
    breaking_coeff:    float = 0.55
    rho_water:         float = 1025.0
    g:                 float = 9.81
    bathy_radius_m:    float = 20_000.0
    bathy_n_steps:     int   = 200
    default_h_deep_m:  float = 20.0
    default_h_point_m: float = 3.0
    fetch_id_offset:   int   = 0   # fetch и points оба 0-based (единая нумерация из GDF)

    def run(
        self,
        points_gdf:         gpd.GeoDataFrame,
        fetch_df:           pd.DataFrame,
        weather_csv_df:     pd.DataFrame,
        normal_field:       str = "normal_azimuth_deg",
        bathymetry_service: Optional[object] = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Запускает расчёт для всех точек.

        Returns
        -------
        daily_gdf   : GeoDataFrame  одна строка на (точку × день)
        summary_gdf : GeoDataFrame  одна строка на точку, агрегат за период
        """
        if "point_id" not in points_gdf.columns:
            points_gdf = points_gdf.copy()
            points_gdf["point_id"] = range(len(points_gdf))

        points_gdf = points_gdf.copy()
        points_gdf["_pid_int"] = points_gdf["point_id"].apply(_normalize_point_id)

        # ── предгруппировка метео по точкам (O(n) вместо O(n²)) ──
        weather = weather_csv_df.copy()
        weather["_pid_int"] = weather["point_id"].apply(_normalize_point_id)
        weather["date"]     = pd.to_datetime(weather["date"])
        weather_by_pid: dict[int, pd.DataFrame] = {
            pid: grp.drop(columns=["_pid_int"])
            for pid, grp in weather.groupby("_pid_int")
        }
        logger.info(f"Метео сгруппировано для {len(weather_by_pid)} точек")

        # ── предпарсинг fetch (один раз для всех точек) ───────────
        fetch_lookup = _build_fetch_lookup_all(fetch_df.copy())

        daily_features:   list[dict] = []
        summary_features: list[dict] = []
        total  = len(points_gdf)
        errors = 0

        for i, (_, pt_row) in enumerate(points_gdf.iterrows()):
            pid_int = int(pt_row["_pid_int"])
            geom    = pt_row.geometry
            normal  = float(pt_row[normal_field])
            orig_id = pt_row["point_id"]

            # метеоряд — из предгруппированного словаря
            wdf = weather_by_pid.get(pid_int)
            if wdf is None or wdf.empty:
                logger.warning(f"[{i}/{total}] pid={pid_int}: нет метеоданных, пропускаем")
                errors += 1
                continue

            # fetch — с учётом сдвига нумерации
            fetch_pid = pid_int + self.fetch_id_offset
            trace_df  = fetch_lookup.get(fetch_pid)
            if trace_df is None:
                # пробуем без сдвига (на случай если нумерация совпадает)
                trace_df = fetch_lookup.get(pid_int)
            if trace_df is None:
                logger.warning(
                    f"[{i}/{total}] pid={pid_int} (fetch_pid={fetch_pid}): "
                    f"нет данных fetch, пропускаем"
                )
                errors += 1
                continue

            try:
                svc = WaveClimateService(
                    trace_df=trace_df,
                    wind_ts_df=wdf,
                    shore_normal_deg=normal,
                    bathymetry_service=bathymetry_service,
                    origin_lon=float(geom.x),
                    origin_lat=float(geom.y),
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
                logger.error(f"[{i}/{total}] pid={pid_int}: ошибка расчёта: {exc}")
                errors += 1
                continue

            if daily.empty:
                logger.warning(f"[{i}/{total}] pid={pid_int}: пустой дневной ряд")
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

            if (i + 1) % 25 == 0 or (i + 1) == total:
                logger.info(f"  Обработано {i + 1}/{total} точек, ошибок: {errors}")

        logger.info(
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

    def export(
        self,
        points_path:          str | Path,
        fetch_csv_path:       str | Path,
        weather_geojson_path: str | Path,
        weather_csv_path:     str | Path,
        daily_output_path:    str | Path,
        summary_output_path:  str | Path,
        normal_field:         str = "normal_azimuth_deg",
        bathymetry_service:   Optional[object] = None,
    ) -> tuple[Path, Path]:
        """Читает входные файлы, запускает расчёт, сохраняет GeoJSON.

        weather_geojson_path : wide-GeoJSON с метеоданными (выход шага 4)
        weather_csv_path     : кэш-CSV (генерируется автоматически из GeoJSON
                               если отсутствует или GeoJSON новее)
        """
        points_gdf = gpd.read_file(points_path)
        fetch_df   = pd.read_csv(fetch_csv_path)

        weather_geojson_path = Path(weather_geojson_path)
        weather_csv_path     = Path(weather_csv_path)

        # Автоматическая (пере)генерация CSV если GeoJSON новее или CSV отсутствует
        need_expand = (
            not weather_csv_path.exists()
            or weather_csv_path.stat().st_size < 10
            or weather_csv_path.stat().st_mtime < weather_geojson_path.stat().st_mtime
        )
        if need_expand:
            _expand_weather_geojson(weather_geojson_path, weather_csv_path)
        else:
            logger.info(f"Weather CSV актуален, загружаем из кэша: {weather_csv_path}")

        weather_df = pd.read_csv(weather_csv_path)

        logger.info(
            f"Входные данные: {len(points_gdf)} точек, "
            f"{len(fetch_df)} fetch-строк, "
            f"{len(weather_df):,} метео-строк"
        )

        daily_gdf, summary_gdf = self.run(
            points_gdf=points_gdf,
            fetch_df=fetch_df,
            weather_csv_df=weather_df,
            normal_field=normal_field,
            bathymetry_service=bathymetry_service,
        )

        daily_output_path   = Path(daily_output_path)
        summary_output_path = Path(summary_output_path)
        daily_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)

        if not daily_gdf.empty:
            daily_gdf.to_file(daily_output_path, driver="GeoJSON")
            logger.success(f"Daily GeoJSON ({len(daily_gdf):,} строк): {daily_output_path}")
        else:
            logger.warning("Дневной GeoJSON пуст — файл не записан")

        if not summary_gdf.empty:
            summary_gdf.to_file(summary_output_path, driver="GeoJSON")
            logger.success(f"Summary GeoJSON ({len(summary_gdf)} точек): {summary_output_path}")
        else:
            logger.warning("Сводный GeoJSON пуст — файл не записан")

        return daily_output_path, summary_output_path
