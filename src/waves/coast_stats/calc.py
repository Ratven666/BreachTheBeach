from __future__ import annotations

"""Расчёт статистик: общая часть для рядов (точки и участки) и обработка группы точек."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .config import SEASONS, StatsConfig
from .mathstats import (
    gumbel_levels, percentile_with_zeros, pettitt, seasonal_concentration, trend, wrap180,
)
from .source import Calendar

SECONDS_PER_YEAR = 86_400.0 * 365.2425


def epoch_masks(years: np.ndarray, cfg: StatsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Две сравниваемые эпохи: заданные в конфигурации либо первая и вторая половины ряда."""
    if cfg.epochs is not None:
        (a0, a1), (b0, b1) = cfg.epochs
        return (years >= a0) & (years <= a1), (years >= b0) & (years <= b1)
    half = len(years) // 2
    index = np.arange(len(years))
    return index < half, index >= len(years) - half


def series_stats(power, storm, cal: Calendar, cfg: StatsConfig, extra=None) -> dict[str, np.ndarray]:
    """Тренды, сдвиги и сезонность для набора рядов.

    power, storm: [n, Y, 12] — средняя мощность (Вт/м) и число штормовых дней по месяцам.
    extra: дополнительные годовые ряды {имя: [n, Y]}.
    """
    energy = power * cal.days                                   # Вт/м * сут
    total = energy.sum(axis=(1, 2))
    annual = energy.sum(axis=2) / cal.days.sum(axis=1)
    series = {"mean": (annual, cal.years), "storm_days": (storm.sum(axis=2), cal.years)}
    series |= {name: (values, cal.years) for name, values in (extra or {}).items()}
    for name, months in SEASONS.items():
        e, d, years = energy[:, :, months], cal.days[:, months], cal.years
        if name == "djf":                                       # декабрь + январь, февраль следующего года
            e = np.concatenate([e[:, :-1, :1], e[:, 1:, 1:]], axis=2)
            d = np.concatenate([d[:-1, :1], d[1:, 1:]], axis=1)
            years = years[1:]
        series[name] = (e.sum(axis=2) / d.sum(axis=1), years)

    out: dict[str, np.ndarray] = {"mean_wm": total / cal.days.sum()}
    for name, (x, years) in series.items():
        slope, _, p = trend(x, years, cfg.prewhiten)
        mean = x.mean(axis=1)
        out[f"{name}_slope"] = slope                            # ед. ряда в год
        out[f"{name}_pct_dec"] = np.where(mean > 0, 1000.0 * slope / np.maximum(mean, 1e-12), np.nan)
        out[f"{name}_p"] = p

    # Межгодовая изменчивость, точка перелома, сравнение эпох.
    mean = annual.mean(axis=1)
    out["cv_pct"] = np.where(mean > 0, 100.0 * annual.std(axis=1, ddof=1) / np.maximum(mean, 1e-12), np.nan)
    k, out["cp_p"] = pettitt(annual)
    out["cp_year"] = cal.years[k]
    first, second = epoch_masks(cal.years, cfg)
    m1, m2 = annual[:, first].mean(axis=1), annual[:, second].mean(axis=1)
    out["epoch_change_pct"] = np.where(m1 > 0, 100.0 * (m2 - m1) / np.maximum(m1, 1e-12), np.nan)
    out["epoch_p"] = mannwhitneyu(annual[:, second], annual[:, first], axis=1).pvalue

    # Сезонность: средний годовой ход, доли сезонов, концентрация.
    climate = energy.sum(axis=1) / cal.days.sum(axis=0)
    share = energy.sum(axis=1) / np.maximum(total, 1e-12)[:, None]
    for m in range(12):
        out[f"m{m + 1:02d}"] = climate[:, m]
    for name, months in SEASONS.items():
        out[f"share_{name}"] = 100.0 * share[:, months].sum(axis=1)
    out["peak_month"] = climate.argmax(axis=1) + 1
    out["season_conc"], out["season_center"] = seasonal_concentration(share)
    out["storm_peak_month"] = storm.sum(axis=1).argmax(axis=1) + 1
    return out


def process_chunk(rows: np.ndarray, ids: np.ndarray, sections: np.ndarray, n_sections: int,
                  cal: Calendar, cfg: StatsConfig):
    """Статистики группы точек. rows — суточные записи (point_id, day, wind, cwef), ids отсортированы.

    Возвращает таблицу по точкам, суммы по участкам [n_sections, Y, 12] (мощность, штормовые дни)
    и, при save_point_annual, годовые ряды точек.
    """
    P, Y = len(ids), len(cal.years)
    pi = np.searchsorted(ids, rows["p"])
    ym = cal.ym[rows["d"] - cal.start]
    cell, year, c = pi * Y * 12 + ym, pi * Y + ym // 12, rows["c"]

    def monthly(mask=slice(None), weights=None):
        return np.bincount(cell[mask], weights=weights, minlength=P * Y * 12).reshape(P, Y, 12)

    def yearly(weights):
        return np.bincount(year, weights=weights, minlength=P * Y).reshape(P, Y)

    # Перцентили по всем дням (нули восстановлены) и порог шторма каждой точки.
    bounds = np.searchsorted(pi, np.arange(P + 1))
    quant = np.array([
        percentile_with_zeros(np.sort(c[bounds[k]:bounds[k + 1]]), cal.n_days,
                              (cfg.storm_percentile, 95.0, 99.0))
        for k in range(P)
    ])
    is_storm = c > quant[pi, 0]

    energy, active, storm = monthly(weights=c), monthly(), monthly(is_storm)
    power = energy / cal.days
    annual_max = np.zeros(P * Y)
    np.maximum.at(annual_max, year, c)
    annual_max = annual_max.reshape(P, Y)

    out = series_stats(power, storm, cal, cfg,
                       extra={"max": annual_max, "active_days": active.sum(axis=2)})
    total = energy.sum(axis=(1, 2))
    out["energy_gj_m_yr"] = out["mean_wm"] * SECONDS_PER_YEAR / 1e9
    out["active_pct"] = 100.0 * active.sum(axis=(1, 2)) / cal.n_days
    out["storm_thr_wm"], out["p95_wm"], out["p99_wm"] = quant.T
    out["max_wm"] = annual_max.max(axis=1)
    out["storm_days_yr"] = storm.sum(axis=(1, 2)) / Y
    out["storm_energy_pct"] = 100.0 * np.bincount(pi[is_storm], weights=c[is_storm], minlength=P) / np.maximum(total, 1e-12)

    # Штормовые события: серии подряд идущих штормовых дней.
    sp, sd = pi[is_storm], rows["d"][is_storm]
    starts = np.r_[True, (np.diff(sp) != 0) | (np.diff(sd) != 1)] if sp.size else np.zeros(0, bool)
    length = np.bincount(np.cumsum(starts) - 1) if sp.size else np.zeros(0)
    n_events = np.bincount(sp[starts], minlength=P)
    longest = np.zeros(P)
    np.maximum.at(longest, sp[starts], length)
    out["storm_events_yr"] = n_events / Y
    out["storm_dur_mean"] = storm.sum(axis=(1, 2)) / np.maximum(n_events, 1)
    out["storm_dur_max"] = longest

    # Направление подхода волн, взвешенное по энергии; поворот между эпохами.
    angle = np.radians(rows["w"])
    east, north = yearly(c * np.sin(angle)), yearly(c * np.cos(angle))
    out["dir_mean_deg"] = np.degrees(np.arctan2(east.sum(axis=1), north.sum(axis=1))) % 360.0
    out["dir_conc"] = np.hypot(east.sum(axis=1), north.sum(axis=1)) / np.maximum(total, 1e-12)
    first, second = epoch_masks(cal.years, cfg)
    direction = [np.degrees(np.arctan2(east[:, m].sum(axis=1), north[:, m].sum(axis=1))) for m in (first, second)]
    out["dir_shift_deg"] = wrap180(direction[1] - direction[0])

    for period, level in gumbel_levels(annual_max, cfg.return_periods).items():
        out[f"rl{period}_wm"] = level

    table = pd.DataFrame({"point_id": ids, "section_id": sections, **out})

    sec_power, sec_storm = np.zeros((n_sections, Y, 12)), np.zeros((n_sections, Y, 12))
    np.add.at(sec_power, sections, power)
    np.add.at(sec_storm, sections, storm)

    annual = None
    if cfg.save_point_annual:
        annual = pd.DataFrame({
            "point_id": np.repeat(ids, Y), "year": np.tile(cal.years, P),
            "mean_wm": (energy.sum(axis=2) / cal.days.sum(axis=1)).ravel(),
            "max_wm": annual_max.ravel(), "storm_days": storm.sum(axis=2).ravel().astype(int),
            "active_days": active.sum(axis=2).ravel().astype(int),
        })
    return table, sec_power, sec_storm, annual
