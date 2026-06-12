"""Шаг 8 пайплайна — расчёт волнового воздействия на берег.

Запускается после:
    1–3: подготовка береговой линии, точек, нормалей
    4:   метеоданные (wide-GeoJSON → нужна конвертация, см. ниже)
    5–6: слияние линий и трассировка разгонов
    7:   батиметрия (опционально)

Выходные файлы:
    wave_impact_daily.geojson   — одна строка на (точку × день)
    wave_impact_summary.geojson — одна строка на точку, агрегат за период
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.waves.services.wave_climate_batch import WaveClimateBatchProcessor

# ─── Пути ──────────────────────────────────────────────────────────────────
BASE = Path("../nvrsk_calc")

POINTS_PATH    = BASE / "nvrsk_equal_radius_200m_points_with_normals.geojson"
FETCH_CSV      = BASE / "fetch" / "fetch_combined.csv"
WEATHER_GEOJSON = BASE / "normal_points_with_weather.geojson"
WEATHER_CSV    = BASE / "weather_point_timeseries.csv"   # длинный формат

DAILY_OUT   = BASE / "wave_impact_daily.geojson"
SUMMARY_OUT = BASE / "wave_impact_summary.geojson"

# ─── Параметры расчёта ─────────────────────────────────────────────────────
DEFAULT_H_DEEP_M  = 20.0   # глубина в открытом море (без батиметрии)
DEFAULT_H_POINT_M = 3.0    # глубина у берега (без батиметрии)
OVERWATER_FACTOR  = 1.1    # поправочный коэффициент скорости ветра над водой
BREAKING_COEFF    = 0.55   # коэффициент обрушения волн γ_b


def expand_weather_geojson(src: Path, dst: Path) -> None:
    """Разворачивает wide-GeoJSON метеоданных в длинный CSV.

    WeatherLayerWrapper сохраняет каждую точку как Feature с properties вида::

        {
          "point_id": 1,
          "date": ["2020-01-01", "2020-01-02", ...],
          "wind_speed_10m_max": [15.3, 8.1, ...],
          "wind_direction_10m_dominant": [270, 180, ...]
        }

    Функция разворачивает это в таблицу:
        point_id | date | wind_speed_10m_max | wind_direction_10m_dominant
    """
    if dst.exists():
        logger.info(f"Weather CSV already exists, skipping expand: {dst}")
        return

    gdf = gpd.read_file(src)
    rows: list[dict] = []

    for _, feat in gdf.iterrows():
        pid = feat.get("point_id") if "point_id" in feat.index else int(feat.name) + 1

        dates_raw = feat.get("date") or feat.get("dates")
        ws_raw    = feat.get("wind_speed_10m_max") or feat.get("ws_kmh")
        wd_raw    = (
            feat.get("wind_direction_10m_dominant")
            or feat.get("wind_dir")
            or feat.get("direction")
        )

        # При read_file geopandas JSON-списки превращаются в строки
        if isinstance(dates_raw, str):
            dates_raw = json.loads(dates_raw)
        if isinstance(ws_raw, str):
            ws_raw = json.loads(ws_raw)
        if isinstance(wd_raw, str):
            wd_raw = json.loads(wd_raw)

        if not (dates_raw and ws_raw and wd_raw):
            logger.warning(f"Point {pid}: missing weather columns, skipping")
            continue

        for d, ws, wd in zip(dates_raw, ws_raw, wd_raw):
            rows.append({
                "point_id": pid,
                "date": d,
                "wind_speed_10m_max": ws,
                "wind_direction_10m_dominant": wd,
            })

    df = pd.DataFrame(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    logger.success(f"Weather CSV saved ({len(df):,} rows): {dst}")


def main() -> None:
    # Шаг 1: конвертация wide-GeoJSON → длинный CSV (если нужно)
    if not WEATHER_CSV.exists():
        logger.info("Converting weather GeoJSON to long CSV...")
        expand_weather_geojson(WEATHER_GEOJSON, WEATHER_CSV)

    # Шаг 2: батч-расчёт волнового воздействия
    processor = WaveClimateBatchProcessor(
        default_h_deep_m=DEFAULT_H_DEEP_M,
        default_h_point_m=DEFAULT_H_POINT_M,
        overwater_factor=OVERWATER_FACTOR,
        breaking_coeff=BREAKING_COEFF,
    )

    logger.info("Starting wave impact batch calculation...")
    logger.info(f"  Points:  {POINTS_PATH}")
    logger.info(f"  Fetch:   {FETCH_CSV}")
    logger.info(f"  Weather: {WEATHER_CSV}")

    daily_path, summary_path = processor.export(
        points_path=POINTS_PATH,
        fetch_csv_path=FETCH_CSV,
        weather_csv_path=WEATHER_CSV,
        daily_output_path=DAILY_OUT,
        summary_output_path=SUMMARY_OUT,
    )

    logger.success(f"Daily GeoJSON:   {daily_path}")
    logger.success(f"Summary GeoJSON: {summary_path}")


if __name__ == "__main__":
    main()
