from __future__ import annotations

"""Сборка БД статистики волновой активности по береговым точкам и участкам."""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from .calc import process_chunk, series_stats
from .config import ALL_SECTION, METRICS, SECTION_METRICS, StatsConfig
from .mathstats import bh_fdr, classify
from .source import assign_sections, load_calendar, load_points, make_engine, read_activity

_context = None  # (engine, cfg, calendar, n_sections) — своё соединение в каждом процессе


def _init_worker(cfg: StatsConfig, calendar, n_sections: int) -> None:
    global _context
    _context = (make_engine(cfg.activity_db, readonly=True), cfg, calendar, n_sections)


def _run_chunk(task):
    ids, sections = task
    engine, cfg, calendar, n_sections = _context
    return process_chunk(read_activity(engine, ids, calendar), ids, sections, n_sections, calendar, cfg)


def _add_classes(table: pd.DataFrame, metrics, cfg: StatsConfig, rows=slice(None)) -> None:
    """Поправка FDR по строкам rows и класс изменения для каждого ряда."""
    for name in metrics:
        q = table[f"{name}_p"].to_numpy().copy()
        q[rows] = bh_fdr(q[rows])
        table[f"{name}_q"] = q
        table[f"{name}_class"] = classify(table[f"{name}_slope"], table[f"{name}_pct_dec"], q,
                                          cfg.alpha, cfg.min_pct_per_decade, cfg.strong_pct_per_decade)


def _circular_mean_lon(lon: pd.Series) -> float:
    angle = np.radians(lon)
    return float(np.degrees(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())))


def _section_tables(points, names, sec_power, sec_storm, cal, cfg):
    """Статистики участков по осреднённым рядам + доли точек по классам изменения."""
    count = np.bincount(points["section_id"], minlength=len(names)).astype(float)
    # Последняя строка — все точки вместе.
    power = np.concatenate([sec_power, sec_power.sum(axis=0, keepdims=True)]) / np.r_[count, count.sum()][:, None, None]
    storm = np.concatenate([sec_storm, sec_storm.sum(axis=0, keepdims=True)]) / np.r_[count, count.sum()][:, None, None]
    ids = np.r_[np.arange(len(names)), -1]

    table = pd.DataFrame({"section_id": ids, "section": [*names, ALL_SECTION],
                          "n_points": np.r_[count, count.sum()].astype(int),
                          **series_stats(power, storm, cal, cfg)})
    _add_classes(table, SECTION_METRICS, cfg, rows=slice(0, len(names)))

    groups = [(i, g) for i, g in points.groupby("section_id")] + [(-1, points)]
    shares = pd.DataFrame([{
        "section_id": i, "lon": _circular_mean_lon(g["lon"]), "lat": g["lat"].mean(),
        "pct_up": 100.0 * (g["mean_class"] > 0).mean(), "pct_down": 100.0 * (g["mean_class"] < 0).mean(),
        "pct_stable": 100.0 * (g["mean_class"] == 0).mean(),
        "point_pct_dec_median": g["mean_pct_dec"].median(),
        "point_mean_wm_median": g["mean_wm"].median(),
        "storm_days_up_pct": 100.0 * (g["storm_days_class"] > 0).mean(),
        "storm_days_down_pct": 100.0 * (g["storm_days_class"] < 0).mean(),
    } for i, g in groups])
    table = table.merge(shares, on="section_id")

    Y = len(cal.years)
    monthly = pd.DataFrame({
        "section_id": np.repeat(ids, Y * 12), "year": np.tile(np.repeat(cal.years, 12), len(ids)),
        "month": np.tile(np.arange(1, 13), len(ids) * Y),
        "mean_wm": power.ravel(), "storm_days": storm.ravel(),
    })
    return table, monthly


def build_coast_stats(cfg: StatsConfig) -> dict:
    """Полный расчёт: точки -> участки -> БД результатов. Возвращает краткую сводку."""
    started = time.monotonic()
    source = make_engine(cfg.activity_db, readonly=True)
    points = load_points(source, cfg.points_geojson, cfg.match_tolerance_m)
    cal = load_calendar(source, cfg.year_from, cfg.year_to)
    section_ids, names = assign_sections(points, cfg)
    ids = points["point_id"].to_numpy()
    logger.info("Точек: {}; участков: {}; период: {}–{} ({} лет)",
                len(ids), len(names), cal.years[0], cal.years[-1], len(cal.years))

    tasks = [(ids[i:i + cfg.chunk_points], section_ids[i:i + cfg.chunk_points])
             for i in range(0, len(ids), cfg.chunk_points)]
    output = make_engine(cfg.output_db)
    tables, done = [], 0
    sec_power = sec_storm = 0.0

    def consume(results):
        nonlocal sec_power, sec_storm, done
        for k, (table, power, storm, annual) in enumerate(results, start=1):
            tables.append(table)
            sec_power, sec_storm, done = sec_power + power, sec_storm + storm, done + len(table)
            if annual is not None:
                annual.round(3).to_sql("point_annual", output, index=False,
                                       if_exists="replace" if k == 1 else "append")
            if k % 25 == 0 or k == len(tasks):
                logger.info("Обработано точек: {}/{} ({:.0f} с)", done, len(ids), time.monotonic() - started)

    args = (cfg, cal, len(names))
    if cfg.n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(cfg.n_workers, initializer=_init_worker, initargs=args) as pool:
            consume(pool.map(_run_chunk, tasks))
    else:
        _init_worker(*args)
        consume(map(_run_chunk, tasks))

    stats = pd.concat(tables, ignore_index=True)
    _add_classes(stats, METRICS, cfg)
    result = points.merge(stats, on="point_id")
    result.insert(5, "section", np.asarray(names, dtype=object)[result["section_id"]])
    sections, monthly = _section_tables(result, names, sec_power, sec_storm, cal, cfg)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "year_from": int(cal.years[0]), "year_to": int(cal.years[-1]), "n_points": len(result),
        "config": json.dumps(asdict(cfg), default=str, ensure_ascii=False),
    }
    result.round(5).to_sql("point_stats", output, index=False, if_exists="replace", chunksize=10_000)
    sections.round(5).to_sql("section_stats", output, index=False, if_exists="replace")
    monthly.round(4).to_sql("section_monthly", output, index=False, if_exists="replace", chunksize=10_000)
    pd.DataFrame({"key": list(meta), "value": [str(v) for v in meta.values()]}).to_sql(
        "meta", output, index=False, if_exists="replace")

    counts = result["mean_class"].value_counts()
    summary = {
        "n_points": len(result), "n_sections": len(names),
        "up": int(counts[counts.index > 0].sum()), "down": int(counts[counts.index < 0].sum()),
        "stable": int(counts.get(0, 0)), "seconds": round(time.monotonic() - started, 1),
    }
    logger.success("Статистика записана: {} | рост: {up}, снижение: {down}, без изменений: {stable}",
                   cfg.output_db, **summary)
    return summary
