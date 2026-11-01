"""
Демонстрация IDW-интерполяции метеоданных.

Точка запроса (QUERY_LAT, QUERY_LON) намеренно расположена МЕЖДУ узлами
сетки (шаг 0.25°), чтобы показать реальную интерполяцию, а не чтение
исходного узла.

Запуск:
    python db_pypeline/0_3_demo_interpolated_weather_archive.py
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from src.weather_history.archive import (
    CompactWeatherRepository,
    GridPoint,
    InterpolatedWeatherRecord,
)

# ------------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------------

DB_PATH = Path("data/db/weather_compact.db")

# Точка между узлами сетки (шаг 0.25°).
# 42.625 и 130.625 — серединное смещение между 42.50/42.75 и 130.50/130.75.
QUERY_LAT = 42.625
QUERY_LON = 130.625

QUERY_DATE = date(2015, 1, 1)

PERIOD_START = date(2005, 1, 1)
PERIOD_END = date(2015, 12, 31)

_WIND_COLS = ["wind_speed_max", "wind_speed_mean", "wind_gust_max", "wind_direction"]

# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------


def _records_to_df(records: list[InterpolatedWeatherRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_WIND_COLS)

    rows = [
        {
            "obs_date": r.obs_date,
            **{col: getattr(r, col, None) for col in _WIND_COLS},
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df.set_index("obs_date")


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _angular_diff(a: float | None, b: float | None) -> float | None:
    """Минимальная разность двух углов в диапазоне [0, 180]."""
    if a is None or b is None:
        return None
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _print_neighbour_table(
    neighbours: list[GridPoint], lat: float, lon: float
) -> None:
    print(f"  {'ID':>4}  {'lat':>8}  {'lon':>9}  {'dist_deg':>10}  {'IDW w':>12}")
    print(f"  {'—'*4}  {'—'*8}  {'—'*9}  {'—'*10}  {'—'*12}")
    for pt in neighbours:
        dist = math.sqrt((lat - pt.lat) ** 2 + (lon - pt.lon) ** 2)
        w = 1.0 / dist**2 if dist > 1e-9 else float("inf")
        print(f"  {pt.id:>4}  {pt.lat:>8.4f}  {pt.lon:>9.4f}  {dist:>10.6f}  {w:>12.4f}")


def _compare_nearest_vs_idw(
    df_nearest: pd.DataFrame, df_idw: pd.DataFrame
) -> None:
    common = df_nearest.index.intersection(df_idw.index)
    if common.empty:
        print("  Нет общих дат для сравнения.")
        return
    print(f"\n  Общих дат: {len(common)}\n")
    print(f"  {'Поле':<22}  {'MAE':>10}  {'Макс. откл.':>14}")
    print(f"  {'—'*22}  {'—'*10}  {'—'*14}")
    for col in _WIND_COLS:
        if col not in df_nearest.columns or col not in df_idw.columns:
            continue
        a = df_nearest.loc[common, col].dropna()
        b = df_idw.loc[common, col].dropna()
        shared = a.index.intersection(b.index)
        if shared.empty:
            print(f"  {col:<22}  {'—':>10}  {'—':>14}")
            continue
        if col == "wind_direction":
            diff = (a.loc[shared] - b.loc[shared]).apply(
                lambda d: min(abs(d) % 360, 360 - abs(d) % 360)
            )
        else:
            diff = (a.loc[shared] - b.loc[shared]).abs()
        print(f"  {col:<22}  {diff.mean():>10.4f}  {diff.max():>14.4f}")


# ------------------------------------------------------------------
# Основная логика
# ------------------------------------------------------------------


def main() -> None:
    with CompactWeatherRepository(DB_PATH) as repo:

        _print_section("Информация об архиве")
        s = repo.info()
        print(f"  Точек: {s['point_count']}, дней: {s['day_count']}")
        print(f"  Период: {s['date_min']} — {s['date_max']}")

        _print_section(f"Ближайший узел для ({QUERY_LAT}, {QUERY_LON})")
        nearest = repo.nearest_grid_point(QUERY_LAT, QUERY_LON)
        if nearest is None:
            print("  Узел не найден. Завершение.")
            return
        print(f"  {nearest}")
        dist_to_nearest = math.sqrt(
            (QUERY_LAT - nearest.lat) ** 2 + (QUERY_LON - nearest.lon) ** 2
        )
        print(f"  Расстояние до ближайшего узла: {dist_to_nearest:.6f}°")
        if dist_to_nearest < 1e-9:
            print(
                "  ⚠ Точка запроса совпадает с узлом сетки! "
                "Увеличьте смещение QUERY_LAT/QUERY_LON."
            )

        _print_section("Окрестность 3×3 и IDW-веса")
        neighbours = repo._neighbourhood_3x3(QUERY_LAT, QUERY_LON)
        print(f"  Найдено узлов: {len(neighbours)}")
        _print_neighbour_table(neighbours, QUERY_LAT, QUERY_LON)

        assert len(neighbours) >= 2, (
            "Слишком мало соседей — убедитесь, что архив содержит "
            "данные вблизи точки запроса."
        )

        _print_section(f"Запись на {QUERY_DATE}: ближайший узел vs IDW")
        rec_nearest = repo.get_record_for_point(nearest, QUERY_DATE)
        rec_idw_list = repo.get_interpolated_timeseries(
            QUERY_LAT, QUERY_LON,
            start_date=QUERY_DATE, end_date=QUERY_DATE,
        )
        rec_idw = rec_idw_list[0] if rec_idw_list else None

        print(f"  {'Поле':<22}  {'Ближайший':>12}  {'IDW':>12}  {'Δ':>10}")
        print(f"  {'—'*22}  {'—'*12}  {'—'*12}  {'—'*10}")
        for col in _WIND_COLS:
            v_near = getattr(rec_nearest, col, None) if rec_nearest else None
            v_idw = getattr(rec_idw, col, None) if rec_idw else None
            if col == "wind_direction":
                delta = _angular_diff(v_near, v_idw)
                delta_str = f"{delta:+.3f}" if delta is not None else "—"
            else:
                delta = (
                    v_idw - v_near
                    if v_near is not None and v_idw is not None
                    else None
                )
                delta_str = f"{delta:+.3f}" if delta is not None else "—"
            near_str = f"{v_near:.3f}" if v_near is not None else "—"
            idw_str = f"{v_idw:.3f}" if v_idw is not None else "—"
            print(f"  {col:<22}  {near_str:>12}  {idw_str:>12}  {delta_str:>10}")

        _print_section(f"Временной ряд: ближайший узел  {PERIOD_START}—{PERIOD_END}")
        df_nearest = repo.get_dataframe_for_point(
            nearest, start_date=PERIOD_START, end_date=PERIOD_END
        )
        print(f"  Записей: {len(df_nearest)}")
        if not df_nearest.empty:
            print()
            print(
                df_nearest[_WIND_COLS].describe().to_string(
                    float_format=lambda x: f"{x:.3f}"
                )
            )

        _print_section(f"IDW-интерполированный ряд  {PERIOD_START}—{PERIOD_END}")
        interp_series = repo.get_interpolated_timeseries(
            QUERY_LAT, QUERY_LON,
            start_date=PERIOD_START, end_date=PERIOD_END,
        )
        df_idw = _records_to_df(interp_series)
        print(f"  Записей: {len(df_idw)}")
        if not df_idw.empty:
            wind_cols_present = [c for c in _WIND_COLS if c in df_idw.columns]
            print()
            print(
                df_idw[wind_cols_present].describe().to_string(
                    float_format=lambda x: f"{x:.3f}"
                )
            )

        if not df_nearest.empty and not df_idw.empty:
            _print_section("MAE: ближайший узел vs IDW")
            _compare_nearest_vs_idw(
                df_nearest[_WIND_COLS],
                df_idw[[c for c in _WIND_COLS if c in df_idw.columns]],
            )

        if not df_idw.empty:
            out_dir = Path("data/output")
            out_dir.mkdir(parents=True, exist_ok=True)
            path_nearest = out_dir / "weather_nearest.csv"
            path_idw = out_dir / "weather_interpolated.csv"
            df_nearest.to_csv(path_nearest)
            df_idw.to_csv(path_idw)
            _print_section("Экспорт CSV")
            print(f"  Ближайший узел   → {path_nearest}")
            print(f"  IDW-интерполяция → {path_idw}")

        print()


if __name__ == "__main__":
    main()
