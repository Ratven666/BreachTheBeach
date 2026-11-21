from __future__ import annotations

"""Отчёт по БД статистики: карты, графики, интерактивный HTML, GeoJSON и текстовые выводы."""

from pathlib import Path
from textwrap import fill

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from plotly.subplots import make_subplots
from pyproj import Transformer

from .config import CLASS_LABELS, METRICS, SEASONS, SECTION_METRICS
from .source import make_engine

MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
MONTH_COLORS = ["#4575b4", "#74add1", "#a6d96a", "#66bd63", "#1a9850", "#fee08b",
                "#fdae61", "#f46d43", "#d73027", "#a50026", "#762a83", "#313695"]
SEASON_COLORS = {"djf": "#4575b4", "mam": "#1a9850", "jja": "#f46d43", "son": "#a50026"}
CLASS_COLORS = {-2: "#2166ac", -1: "#67a9cf", 0: "#c4c4c4", 1: "#ef8a62", 2: "#b2182b"}
CLASS_ORDER = (-2, -1, 0, 1, 2)
MONTH_DAYS = np.array([31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Стили слоёв карт: (палитра, нормировка | None, подписи делений | None).
CLASS_STYLE = (ListedColormap([CLASS_COLORS[k] for k in CLASS_ORDER]), BoundaryNorm(np.arange(-2.5, 3), 5), CLASS_LABELS)
MONTH_STYLE = (ListedColormap(MONTH_COLORS), BoundaryNorm(np.arange(0.5, 13), 12), dict(enumerate(MONTHS, 1)))

# Столбцы point_stats, попадающие в GeoJSON (для QGIS).
GEOJSON_COLUMNS = [
    "point_id", "section", "mean_wm", "energy_gj_m_yr", "mean_pct_dec", "mean_q", "mean_class",
    "storm_days_pct_dec", "storm_days_class", "max_pct_dec", "max_class", "active_days_class",
    "djf_pct_dec", "djf_class", "mam_pct_dec", "mam_class", "jja_pct_dec", "jja_class",
    "son_pct_dec", "son_class", "epoch_change_pct", "epoch_p", "cp_year", "cp_p", "cv_pct",
    "peak_month", "season_conc", "share_djf", "share_mam", "share_jja", "share_son",
    "storm_thr_wm", "storm_days_yr", "storm_energy_pct", "storm_dur_mean", "p99_wm", "rl50_wm",
    "dir_mean_deg", "dir_shift_deg",
]


def _symmetric(values) -> TwoSlopeNorm:
    limit = max(float(np.nanpercentile(np.abs(values), 98)), 1e-6)
    return TwoSlopeNorm(0.0, -limit, limit)


def _save(fig, path: Path, dpi: int) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ==================== КАРТА ====================

class CoastMap:
    """Равновеликая азимутальная проекция с центром в данных (км); без разрыва на 180°."""

    def __init__(self, lon, lat, background_geojson: Path | None = None):
        lon0 = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
        crs = f"+proj=laea +lat_0={lat.mean():.3f} +lon_0={lon0:.3f} +units=km"
        self.project = Transformer.from_crs(4326, crs, always_xy=True).transform
        self.x, self.y = self.project(np.asarray(lon), np.asarray(lat))
        self.land = gpd.read_file(background_geojson).to_crs(crs) if background_geojson else None
        self.size = float(np.clip(40_000 / len(lon), 0.3, 14))
        pad = 0.04 * max(np.ptp(self.x), np.ptp(self.y), 50.0)
        self.xlim = (self.x.min() - pad, self.x.max() + pad)
        self.ylim = (self.y.min() - pad, self.y.max() + pad)
        self.aspect = (self.ylim[1] - self.ylim[0]) / (self.xlim[1] - self.xlim[0])
        self.lat_range = (max(lat.min() - 30, -85), min(lat.max() + 30, 89))

    def panels(self, n_rows: int, n_cols: int, width: float = 16.0):
        panel = width / n_cols
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout="constrained",
                                 figsize=(width, n_rows * (panel * self.aspect * 0.86 + 0.5)))
        return fig, axes.ravel()

    def _graticule(self, ax) -> None:
        span = max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]) / 111.0   # градусы
        step = next(s for s in (1, 2, 5, 10, 20) if span / s <= 8 or s == 20)
        style = dict(color="0.75", lw=0.4, zorder=0)
        text = dict(fontsize=6, color="0.4", clip_on=True)
        for lon in np.arange(-180, 180, 2 * step if step >= 5 else step):
            x, y = self.project(np.full(200, lon), np.linspace(*self.lat_range, 200))
            ax.plot(x, y, **style)
            k = np.flatnonzero(np.diff(y > self.ylim[0]))            # пересечение с нижней рамкой
            if k.size and self.xlim[0] < x[k[0]] < self.xlim[1]:
                ax.text(x[k[0]], self.ylim[0], f" {abs(lon):g}°{'в.д.' if lon >= 0 else 'з.д.'}", va="bottom", **text)
        for lat in np.arange(-80, 90, step):
            x, y = self.project(np.linspace(-180, 180, 721), np.full(721, lat))
            ax.plot(x, y, **style)
            k = np.flatnonzero(np.diff(x > self.xlim[0]))            # пересечение с левой рамкой
            if k.size and self.ylim[0] < y[k[0]] < self.ylim[1]:
                ax.text(self.xlim[0], y[k[0]], f" {lat:g}°", va="bottom", **text)

    def draw(self, ax, values, title: str, style, label: str = ""):
        """Точки, окрашенные по values; нейтральные значения рисуются первыми."""
        cmap, norm, ticks = style
        values = np.asarray(values, dtype=float)
        norm = norm or plt.Normalize(*np.nanpercentile(values, (2, 98)))
        if self.land is not None:
            self.land.plot(ax=ax, color="#f1efe9", edgecolor="0.7", linewidth=0.3, zorder=0)
        self._graticule(ax)
        order = np.argsort(np.abs(np.nan_to_num(values)), kind="stable")
        artist = ax.scatter(self.x[order], self.y[order], c=values[order], s=self.size, cmap=cmap,
                            norm=norm, linewidths=0, rasterized=True, zorder=2)
        bar = ax.figure.colorbar(artist, ax=ax, shrink=0.85, pad=0.01, aspect=30, label=label,
                                 ticks=list(ticks) if ticks else None)
        if ticks:
            bar.ax.set_yticklabels(list(ticks.values()), fontsize=7)
        ax.set(xlim=self.xlim, ylim=self.ylim, aspect="equal", xticks=[], yticks=[])
        ax.set_title(title, fontsize=10, loc="left")
        return artist


# ==================== КАРТЫ ПО ТОЧКАМ ====================

def _class_maps(cmap: CoastMap, points, metrics, n_cols: int, title: str):
    fig, axes = cmap.panels(-(-len(metrics) // n_cols), n_cols)
    for ax, name in zip(axes, metrics):
        cls = points[f"{name}_class"]
        share = f"рост {100 * (cls > 0).mean():.0f} %, снижение {100 * (cls < 0).mean():.0f} %"
        cmap.draw(ax, cls, f"{METRICS[name]}: {share}", CLASS_STYLE)
    fig.suptitle(title, fontsize=12)
    return fig


def _trend_map(cmap: CoastMap, points, period: str):
    fig, axes = cmap.panels(2, 1, width=13)
    cmap.draw(axes[0], points["mean_class"], f"Класс изменения средней мощности, {period}", CLASS_STYLE)
    cmap.draw(axes[1], points["mean_pct_dec"], "Наклон Тейла–Сена средней мощности",
              ("RdBu_r", _symmetric(points["mean_pct_dec"]), None), "% за 10 лет")
    return fig


def _extra_trend_maps(cmap: CoastMap, points):
    fig, axes = cmap.panels(2, 2)
    for ax, name in zip(axes, ("storm_days", "max", "active_days")):
        cmap.draw(ax, points[f"{name}_class"], METRICS[name], CLASS_STYLE)
    cmap.draw(axes[3], points["epoch_change_pct"], "Средняя мощность: вторая эпоха к первой",
              ("RdBu_r", _symmetric(points["epoch_change_pct"]), None), "%")
    return fig


def _seasonality_maps(cmap: CoastMap, points):
    fig, axes = cmap.panels(2, 2)
    cmap.draw(axes[0], points["peak_month"].where(points["mean_wm"] > 0), "Месяц наибольшей средней мощности", MONTH_STYLE)
    cmap.draw(axes[1], points["season_conc"], "Сезонная концентрация энергии (0 — равномерно, 1 — один месяц)",
              ("viridis", None, None))
    cmap.draw(axes[2], points["share_djf"], "Доля зимы (XII–II) в годовой энергии", ("cividis", None, None), "%")
    power = points["mean_wm"].where(points["mean_wm"] > 0)
    cmap.draw(axes[3], power, "Средняя мощность потока волновой энергии",
              ("magma_r", LogNorm(*np.nanpercentile(power, (2, 98))), None), "Вт/м")
    return fig


def _sections_map(cmap: CoastMap, points, sections):
    fig, axes = cmap.panels(1, 1, width=13)
    colors = plt.get_cmap("tab20")(points["section_id"].to_numpy() % 20)
    axes[0].scatter(cmap.x, cmap.y, c=colors, s=cmap.size, linewidths=0, rasterized=True)
    cmap._graticule(axes[0])
    for row in sections.itertuples():
        x, y = cmap.project(row.lon, row.lat)
        axes[0].text(x, y, str(row.section_id), fontsize=7, ha="center", va="center", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", lw=0.3, alpha=0.85))
    axes[0].set(xlim=cmap.xlim, ylim=cmap.ylim, aspect="equal", xticks=[], yticks=[])
    axes[0].set_title("Участки побережья (номера — section_id)", fontsize=10, loc="left")
    return fig


# ==================== ВРЕМЕННЫЕ РЯДЫ ====================

def _monthly_cube(monthly: pd.DataFrame, section_id: int):
    """Ряды участка: годы, мощность [Y, 12], число дней [Y, 12], штормовые дни [Y, 12]."""
    part = monthly[monthly["section_id"] == section_id]
    power = part.pivot(index="year", columns="month", values="mean_wm")
    storm = part.pivot(index="year", columns="month", values="storm_days").to_numpy()
    years = power.index.to_numpy()
    days = np.array([[pd.Period(year=y, month=m, freq="M").days_in_month for m in range(1, 13)] for y in years])
    return years, power.to_numpy(), days, storm


def _season_series(power, days, years, name: str):
    """Средняя мощность сезона по годам; зима — декабрь предыдущего года + январь, февраль."""
    months = list(SEASONS[name])
    e, d = (power * days)[:, months], days[:, months]
    if name == "djf":
        e, d, years = np.c_[e[:-1, :1], e[1:, 1:]], np.c_[d[:-1, :1], d[1:, 1:]], years[1:]
    return years, e.sum(axis=1) / d.sum(axis=1)


def _sen_line(years, values, slope):
    return np.median(values - slope * (years - years[0])) + slope * (years - years[0])


def _star(row, name: str) -> str:
    return "*" if row[f"{name}_class"] != 0 else ""


def _time_figure(monthly, row, title: str):
    years, power, days, storm = _monthly_cube(monthly, row["section_id"])
    annual = (power * days).sum(axis=1) / days.sum(axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), layout="constrained")

    ax = axes[0, 0]
    ax.bar(years, annual, color="#9ecae1", width=0.85, label="год")
    ax.plot(years, pd.Series(annual).rolling(5, center=True).mean(), color="#08519c", lw=2, label="скользящее среднее, 5 лет")
    ax.plot(years, _sen_line(years, annual, row["mean_slope"]), color="#b2182b", lw=1.5, ls="--",
            label=f"тренд Сена: {row['mean_pct_dec']:+.2f} % за 10 лет (p = {row['mean_p']:.3f})")
    ax.axvline(row["cp_year"] - 0.5, color="0.3", lw=0.8, ls=":", label=f"сдвиг (Петтитт): {row['cp_year']:.0f}, p = {row['cp_p']:.3f}")
    ax.set(ylim=(annual.min() * 0.9, annual.max() * 1.08), ylabel="Вт/м", title="Среднегодовая мощность")
    ax.legend(fontsize=8, loc="upper left", ncols=2)

    ax = axes[0, 1]
    for name, color in SEASON_COLORS.items():
        t, values = _season_series(power, days, years, name)
        ax.plot(t, values, color=color, lw=0.6, alpha=0.45)
        ax.plot(t, pd.Series(values).rolling(5, center=True).mean(), color=color, lw=2,
                label=f"{METRICS[name]}: {row[f'{name}_pct_dec']:+.2f} % / 10 лет{_star(row, name)}")
    ax.set(ylabel="Вт/м", title="Средняя мощность по сезонам (жирная линия — среднее за 5 лет; * — значимо)")
    ax.legend(fontsize=8, ncols=2)

    ax = axes[1, 0]
    climate = power.mean(axis=0)
    anomaly = 100.0 * (power - climate) / np.maximum(climate, 1e-9)
    mesh = ax.pcolormesh(years, np.arange(1, 13), anomaly.T, cmap="RdBu_r", norm=_symmetric(anomaly), shading="nearest")
    ax.set(yticks=range(1, 13), yticklabels=MONTHS, title="Отклонение месячной мощности от нормы месяца")
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax, pad=0.01, label="%")

    ax = axes[1, 1]
    x = np.arange(1, 13)
    ax.fill_between(x, power.min(axis=0), power.max(axis=0), color="#deebf7", label="мин–макс по годам")
    ax.fill_between(x, *np.percentile(power, (25, 75), axis=0), color="#9ecae1", label="25–75 %")
    ax.plot(x, climate, color="#08519c", lw=2, marker="o", ms=4, label="средняя")
    twin = ax.twinx()
    twin.bar(x, storm.mean(axis=0), width=0.35, color="#b2182b", alpha=0.55, label="штормовых дней в месяц")
    twin.set_ylabel("штормовых дней")
    ax.set(xticks=x, xticklabels=MONTHS, ylabel="Вт/м", ylim=(None, power.max() * 1.15),
           title="Годовой ход: мощность и штормовые дни")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=8, loc="upper center", ncols=4)
    fig.suptitle(title, fontsize=13)
    return fig


# ==================== УЧАСТКИ ====================

def _section_labels(sections) -> list[str]:
    return [f"{r.section_id}: {r.section} ({r.n_points})" for r in sections.itertuples()]


def _sections_trend(sections):
    n = len(sections)
    y = np.arange(n)
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, max(4.5, 0.27 * n + 1.8)), layout="constrained",
                                      width_ratios=(1, 1.25), sharey=True)
    start = np.zeros(n)
    for column, color, label in (("pct_down", CLASS_COLORS[-2], "снижение"), ("pct_stable", CLASS_COLORS[0], "без изменений"),
                                 ("pct_up", CLASS_COLORS[2], "рост")):
        left.barh(y, sections[column], left=start, color=color, label=label, height=0.8)
        start += sections[column].to_numpy()
    left.set(yticks=y, xlim=(0, 100), xlabel="% точек участка", title="Доли точек по характеру изменения")
    left.set_yticklabels(_section_labels(sections), fontsize=7)
    left.legend(fontsize=8, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.03))
    left.invert_yaxis()

    values = sections[[f"{m}_pct_dec" for m in SECTION_METRICS]].to_numpy()
    mesh = right.pcolormesh(np.arange(len(SECTION_METRICS)), y, values, cmap="RdBu_r", norm=_symmetric(values), shading="nearest")
    for i in range(n):
        for j, name in enumerate(SECTION_METRICS):
            mark = "*" if sections[f"{name}_class"].iloc[i] != 0 else ""
            label = f"{values[i, j]:+.1f}{mark}" if np.isfinite(values[i, j]) else "—"
            right.text(j, i, label, ha="center", va="center", fontsize=6.5)
    right.set_xticks(range(len(SECTION_METRICS)), [fill(METRICS[m], 14) for m in SECTION_METRICS], fontsize=7)
    right.set_title("Тренд осреднённого ряда участка, % за 10 лет (* — значимо)")
    fig.colorbar(mesh, ax=right, pad=0.01, label="% за 10 лет")
    return fig


def _sections_seasonality(sections, monthly):
    n = len(sections)
    y = np.arange(n)
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, max(4.5, 0.27 * n + 1.8)), layout="constrained",
                                      width_ratios=(1, 1.6), sharey=True)
    energy = sections[[f"m{m:02d}" for m in range(1, 13)]].to_numpy() * MONTH_DAYS
    share = 100.0 * energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-9)
    mesh = left.pcolormesh(np.arange(12), y, share, cmap="YlOrRd", shading="nearest")
    left.plot(share.argmax(axis=1), y, "k.", ms=4)
    left.set_xticks(range(12), MONTHS, fontsize=7)
    left.set_yticks(y, _section_labels(sections), fontsize=7)
    left.set_title("Доля месяца в годовой энергии (точка — максимум)")
    left.invert_yaxis()
    fig.colorbar(mesh, ax=left, pad=0.01, label="%")

    ids = sections["section_id"].to_numpy()
    part = monthly[monthly["section_id"] >= 0].copy()
    part["energy"] = part["mean_wm"] * MONTH_DAYS[part["month"] - 1]
    annual = part.pivot_table(index="section_id", columns="year", values="energy", aggfunc="sum").loc[ids]
    z = annual.sub(annual.mean(axis=1), axis=0).div(annual.std(axis=1).replace(0, np.nan), axis=0)
    mesh = right.pcolormesh(annual.columns, y, z.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, shading="nearest")
    right.set_title("Стандартизованные аномалии среднегодовой мощности")
    fig.colorbar(mesh, ax=right, pad=0.01, label="σ")
    return fig


# ==================== РАСПРЕДЕЛЕНИЯ ====================

def _distributions(points):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    ax = axes[0, 0]
    limit = np.nanpercentile(np.abs(points["mean_pct_dec"]), 99.5)
    groups = [points.loc[points["mean_class"] == k, "mean_pct_dec"].dropna() for k in CLASS_ORDER]
    ax.hist(groups, bins=np.linspace(-limit, limit, 61), stacked=True, color=[CLASS_COLORS[k] for k in CLASS_ORDER],
            label=[CLASS_LABELS[k] for k in CLASS_ORDER])
    ax.axvline(points["mean_pct_dec"].median(), color="k", lw=1, ls="--", label="медиана")
    ax.set(xlabel="% за 10 лет", ylabel="точек", title="Тренд средней мощности по точкам")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    start = np.zeros(len(METRICS))
    for k in CLASS_ORDER:
        share = np.array([100.0 * (points[f"{m}_class"] == k).mean() for m in METRICS])
        ax.barh(list(METRICS.values()), share, left=start, color=CLASS_COLORS[k], label=CLASS_LABELS[k])
        start += share
    ax.set(xlim=(0, 100), xlabel="% точек", title="Характер изменения по показателям")
    ax.invert_yaxis()

    ax = axes[1, 0]
    shifted = points[points["cp_p"] < 0.05]
    years = np.arange(points["cp_year"].min(), points["cp_year"].max() + 2) - 0.5
    ax.hist([shifted.loc[shifted["epoch_change_pct"] < 0, "cp_year"], shifted.loc[shifted["epoch_change_pct"] >= 0, "cp_year"]],
            bins=years, stacked=True, color=[CLASS_COLORS[-2], CLASS_COLORS[2]], label=["после сдвига слабее", "после сдвига сильнее"])
    ax.set(xlabel="год сдвига", ylabel="точек",
           title=f"Год сдвига среднего (тест Петтитта, p < 0,05): {100 * len(shifted) / len(points):.0f} % точек")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    active = points[points["mean_wm"] > 0]
    x = np.arange(1, 13)
    ax.bar(x - 0.2, 100.0 * np.bincount(active["peak_month"], minlength=13)[1:] / len(active), width=0.4,
           color=MONTH_COLORS, label="максимум средней мощности")
    ax.bar(x + 0.2, 100.0 * np.bincount(active["storm_peak_month"], minlength=13)[1:] / len(active), width=0.4,
           color="0.35", label="максимум числа штормовых дней")
    ax.set(xticks=x, xticklabels=MONTHS, ylabel="% точек", title="Месяц сезонного максимума по точкам")
    ax.legend(fontsize=8)
    return fig


# ==================== ТЕКСТОВЫЕ ВЫВОДЫ ====================

def _summary(points, sections, total, meta) -> list[tuple[str, list[str]]]:
    def shares(name: str) -> str:
        cls = points[f"{name}_class"]
        return (f"рост — {100 * (cls > 0).mean():.1f} % точек (сильный — {100 * (cls == 2).mean():.1f} %), "
                f"снижение — {100 * (cls < 0).mean():.1f} % (сильное — {100 * (cls == -2).mean():.1f} %), "
                f"без значимых изменений — {100 * (cls == 0).mean():.1f} %")

    def ranked(table, ascending: bool) -> list[str]:
        rows = table.sort_values("mean_pct_dec", ascending=ascending).head(5)
        return [f"{r.section_id}: {r.section} — {r.mean_pct_dec:+.2f} % за 10 лет, точек с ростом {r.pct_up:.0f} %, "
                f"со снижением {r.pct_down:.0f} %" for r in rows.itertuples()] or ["нет"]

    active = points[points["mean_wm"] > 0]
    peak_season = {name: 100.0 * active["peak_month"].isin([m + 1 for m in months]).mean() for name, months in SEASONS.items()}
    by_conc = sections[sections["mean_wm"] > 0].sort_values("season_conc")
    return [
        ("Общая оценка изменений", [
            f"Период: {meta['year_from']}–{meta['year_to']}; точек: {len(points)}; участков: {len(sections)}.",
            f"Средняя мощность: {shares('mean')}.",
            f"Медианный тренд по точкам: {points['mean_pct_dec'].median():+.2f} % за 10 лет; осреднённый ряд всех точек: "
            f"{total['mean_pct_dec']:+.2f} % за 10 лет (p = {total['mean_p']:.3f}, {CLASS_LABELS[total['mean_class']]}).",
            f"Вторая половина периода относительно первой: {total['epoch_change_pct']:+.1f} % (p = {total['epoch_p']:.3f}); "
            f"год сдвига по Петтитту: {total['cp_year']:.0f} (p = {total['cp_p']:.3f}).",
            f"Штормовые дни: {shares('storm_days')}.",
            f"Годовой максимум: {shares('max')}.",
        ]),
        ("Изменения по сезонам", [
            f"{METRICS[name]}: {shares(name)}; общий ряд {total[f'{name}_pct_dec']:+.2f} % за 10 лет{_star(total, name)}."
            for name in SEASONS
        ]),
        ("Участки с наибольшим значимым ростом", ranked(sections[sections["mean_class"] > 0], False)),
        ("Участки с наибольшим значимым снижением", ranked(sections[sections["mean_class"] < 0], True)),
        ("Сезонность", [
            "Доли сезонов в годовой энергии (все точки): "
            + ", ".join(f"{METRICS[name]} — {total[f'share_{name}']:.1f} %" for name in SEASONS) + ".",
            "Сезон максимума по точкам: " + ", ".join(f"{METRICS[name]} — {value:.1f} %" for name, value in peak_season.items()) + ".",
            f"Сезонная концентрация: медиана по точкам {active['season_conc'].median():.2f}; месяц максимума общего ряда — "
            f"{MONTHS[int(total['peak_month']) - 1]}.",
            "Наиболее выраженная сезонность: " + "; ".join(f"{r.section} ({r.season_conc:.2f})" for r in by_conc.tail(3)[::-1].itertuples()) + ".",
            "Наиболее равномерный годовой ход: " + "; ".join(f"{r.section} ({r.season_conc:.2f})" for r in by_conc.head(3).itertuples()) + ".",
        ]),
    ]


# ==================== ИНТЕРАКТИВНЫЙ HTML ====================

def _discrete_scale(colors) -> list:
    return [[k / len(colors), color] for i, color in enumerate(colors) for k in (i, i + 1)]


def _interactive_map(points: pd.DataFrame, max_points: int) -> go.Figure:
    data = points.sample(max_points, random_state=0) if len(points) > max_points else points
    crossing = np.ptp(data["lon"]) > 180
    lon = data["lon"] % 360 if crossing else data["lon"]
    class_scale = _discrete_scale([CLASS_COLORS[k] for k in CLASS_ORDER])
    # подпись: (значения, шкала, пределы, подписи делений)
    layers = {
        "Класс изменения средней мощности": (data["mean_class"], class_scale, (-2.5, 2.5), CLASS_LABELS),
        "Тренд средней мощности, % / 10 лет": (data["mean_pct_dec"], "RdBu_r", "sym", None),
        "Класс изменения числа штормовых дней": (data["storm_days_class"], class_scale, (-2.5, 2.5), CLASS_LABELS),
        "Тренд зимней мощности, % / 10 лет": (data["djf_pct_dec"], "RdBu_r", "sym", None),
        "Тренд летней мощности, % / 10 лет": (data["jja_pct_dec"], "RdBu_r", "sym", None),
        "Изменение между эпохами, %": (data["epoch_change_pct"], "RdBu_r", "sym", None),
        "Средняя мощность, lg(Вт/м)": (np.log10(data["mean_wm"].clip(lower=1)), "Magma_r", None, None),
        "Месяц максимума": (data["peak_month"], _discrete_scale(MONTH_COLORS), (0.5, 12.5), dict(enumerate(MONTHS, 1))),
        "Сезонная концентрация": (data["season_conc"], "Viridis", None, None),
    }

    def restyle(label: str) -> dict:
        values, scale, limits, ticks = layers[label]
        if limits == "sym":
            limits = (-np.nanpercentile(np.abs(values), 98), np.nanpercentile(np.abs(values), 98))
        low, high = limits or np.nanpercentile(values, (2, 98))
        return {"marker.color": [values.round(3).tolist()], "marker.colorscale": [scale], "marker.cmin": low, "marker.cmax": high,
                "marker.colorbar.tickmode": "array" if ticks else "auto",
                "marker.colorbar.tickvals": [list(ticks) if ticks else None],
                "marker.colorbar.ticktext": [list(ticks.values()) if ticks else None]}

    first = next(iter(layers))
    start = restyle(first)
    custom = np.column_stack([data["point_id"], data["section"], data["mean_wm"].round(1), data["mean_pct_dec"].round(2),
                              data["mean_class"].map(CLASS_LABELS), data["storm_days_pct_dec"].round(2),
                              [MONTHS[m - 1] for m in data["peak_month"]], data["season_conc"].round(2)])
    fig = go.Figure(go.Scattermap(
        lon=lon.round(5), lat=data["lat"].round(5), mode="markers", customdata=custom,
        marker=dict(size=6, color=start["marker.color"][0], colorscale=start["marker.colorscale"][0], cmin=start["marker.cmin"],
                    cmax=start["marker.cmax"], colorbar=dict(thickness=14, tickmode=start["marker.colorbar.tickmode"],
                                                             tickvals=start["marker.colorbar.tickvals"][0],
                                                             ticktext=start["marker.colorbar.ticktext"][0])),
        hovertemplate=("<b>Точка %{customdata[0]}</b><br>%{customdata[1]}<br>средняя мощность: %{customdata[2]} Вт/м<br>"
                       "тренд: %{customdata[3]} % / 10 лет (%{customdata[4]})<br>штормовые дни: %{customdata[5]} % / 10 лет<br>"
                       "максимум: %{customdata[6]}; концентрация: %{customdata[7]}<extra></extra>"),
    ))
    span = max(np.ptp(lon), 2.0 * np.ptp(data["lat"]), 0.01)
    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lon=float(lon.mean()), lat=float(data["lat"].mean())),
                 zoom=float(np.clip(np.log2(360.0 / span) + 0.6, 0.5, 13))),
        height=680, margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="white",
                          buttons=[dict(label=label, method="restyle", args=[restyle(label)]) for label in layers])],
    )
    return fig


def _section_explorer(sections_all: pd.DataFrame, monthly: pd.DataFrame) -> go.Figure:
    """Ряд и календарь «год × месяц» для выбранного участка (первым идёт сводный ряд всех точек)."""
    rows = pd.concat([sections_all[sections_all["section_id"] < 0], sections_all[sections_all["section_id"] >= 0]])
    fig = make_subplots(rows=1, cols=2, column_widths=(0.45, 0.55), horizontal_spacing=0.07,
                        subplot_titles=("Среднегодовая мощность, Вт/м", "Средняя мощность по месяцам, Вт/м"))
    buttons = []
    for k, row in enumerate(rows.to_dict("records")):
        years, power, days, _ = _monthly_cube(monthly, row["section_id"])
        annual = (power * days).sum(axis=1) / days.sum(axis=1)
        show = k == 0
        fig.add_trace(go.Bar(x=years, y=annual.round(2), marker_color="#9ecae1", name="год", visible=show), 1, 1)
        fig.add_trace(go.Scatter(x=years, y=_sen_line(years, annual, row["mean_slope"]).round(2), name="тренд Сена",
                                 line=dict(color="#b2182b", dash="dash"), visible=show), 1, 1)
        fig.add_trace(go.Heatmap(x=years, y=MONTHS, z=power.T.round(1), colorscale="YlOrRd", visible=show,
                                 colorbar=dict(thickness=12)), 1, 2)
        title = (f"{row['section']}: точек {row['n_points']}, тренд {row['mean_pct_dec']:+.2f} % за 10 лет "
                 f"(p = {row['mean_p']:.3f}), рост у {row['pct_up']:.0f} % точек, снижение у {row['pct_down']:.0f} %")
        visible = np.repeat(np.arange(len(rows)) == k, 3).tolist()
        buttons.append(dict(label=f"{row['section_id']}: {row['section']}" if row["section_id"] >= 0 else row["section"],
                            method="update", args=[{"visible": visible}, {"title.text": title}]))
    fig.update_layout(height=520, showlegend=False, title=dict(text=buttons[0]["args"][1]["title.text"], font_size=13, y=0.97),
                      margin=dict(l=50, r=20, t=110, b=40), yaxis2=dict(autorange="reversed"),
                      updatemenus=[dict(x=0.0, y=1.24, xanchor="left", yanchor="top", buttons=buttons)])
    return fig


HTML_STYLE = """
body{font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;margin:0 auto;max-width:1500px;padding:16px 28px;color:#222}
h1{font-size:24px} h2{font-size:19px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px} h3{font-size:15px;margin:16px 0 4px}
li{margin:3px 0;line-height:1.4} img{width:100%;margin:8px 0 18px;border:1px solid #eee} .note{color:#666;font-size:13px}
table{border-collapse:collapse;font-size:12px} th,td{border:1px solid #ddd;padding:3px 7px;text-align:right}
th{background:#f3f3f3;position:sticky;top:0} td:nth-child(2){text-align:left} .scroll{max-height:560px;overflow:auto}
"""

TABLE_COLUMNS = {
    "section_id": "№", "section": "Участок", "n_points": "Точек", "mean_wm": "Мощность, Вт/м",
    "mean_pct_dec": "Тренд, %/10 лет", "mean_q": "q", "pct_up": "Рост, % точек", "pct_down": "Снижение, % точек",
    "storm_days_pct_dec": "Штормовые дни, %/10 лет", "djf_pct_dec": "Зима, %/10 лет", "mam_pct_dec": "Весна, %/10 лет",
    "jja_pct_dec": "Лето, %/10 лет", "son_pct_dec": "Осень, %/10 лет", "epoch_change_pct": "Эпохи, %",
    "cp_year": "Год сдвига", "cv_pct": "CV, %", "peak_month": "Месяц макс.", "season_conc": "Концентрация",
    "share_djf": "Зима, % энергии", "share_jja": "Лето, % энергии",
}


def _write_html(path: Path, title: str, summary, figures: dict[str, Path], map_fig, explorer_fig, sections_all) -> None:
    text = "".join(f"<h3>{head}</h3><ul>{''.join(f'<li>{line}</li>' for line in lines)}</ul>" for head, lines in summary)
    images = "".join(f"<h3>{caption}</h3><img src='{file.name}' loading='lazy'>" for caption, file in figures.items())
    table = sections_all[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS).to_html(index=False, float_format="%.2f", border=0)
    path.write_text(
        f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>{title}</title><style>{HTML_STYLE}</style></head><body>"
        f"<h1>{title}</h1><h2>Выводы</h2>{text}"
        f"<h2>Интерактивная карта точек</h2><p class='note'>Показатель выбирается в списке слева вверху.</p>"
        f"{map_fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        f"<h2>Динамика по участкам</h2>{explorer_fig.to_html(full_html=False, include_plotlyjs=False)}"
        f"<h2>Таблица участков</h2><div class='scroll'>{table}</div>"
        f"<h2>Карты и графики</h2>{images}</body></html>",
        encoding="utf-8",
    )


# ==================== СБОРКА ОТЧЁТА ====================

def build_report(stats_db: Path | str, out_dir: Path, *, background_geojson: Path | None = None,
                 max_html_points: int = 50_000, dpi: int = 150) -> dict[str, Path]:
    """Строит отчёт по БД статистики. background_geojson — необязательная подложка суши."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(stats_db)
    points = pd.read_sql("point_stats", engine)
    sections_all = pd.read_sql("section_stats", engine)
    monthly = pd.read_sql("section_monthly", engine)
    meta = pd.read_sql("meta", engine).set_index("key")["value"]
    sections = sections_all[sections_all["section_id"] >= 0].reset_index(drop=True)
    total = sections_all[sections_all["section_id"] < 0].iloc[0]
    period = f"{meta['year_from']}–{meta['year_to']}"
    cmap = CoastMap(points["lon"], points["lat"], background_geojson)

    plots = {
        "Тренд средней мощности": ("01_trend_map.png", lambda: _trend_map(cmap, points, period)),
        "Тренды по сезонам": ("02_trend_seasons.png", lambda: _class_maps(cmap, points, list(SEASONS), 2, f"Изменение сезонной мощности, {period}")),
        "Штормы, максимумы, активные дни, сравнение эпох": ("03_trend_storms.png", lambda: _extra_trend_maps(cmap, points)),
        "Сезонность": ("04_seasonality_maps.png", lambda: _seasonality_maps(cmap, points)),
        "Все точки: изменение во времени": ("05_overall_time.png", lambda: _time_figure(monthly, total, f"Все точки, {period}")),
        "Участки: тренды": ("06_sections_trend.png", lambda: _sections_trend(sections)),
        "Участки: сезонность и аномалии по годам": ("07_sections_seasonality.png", lambda: _sections_seasonality(sections, monthly)),
        "Распределения по точкам": ("08_distributions.png", lambda: _distributions(points)),
        "Схема участков": ("00_sections_map.png", lambda: _sections_map(cmap, points, sections)),
    }
    figures = {}
    for caption, (name, make) in plots.items():
        figures[caption] = _save(make(), out_dir / name, dpi)
        logger.info("Рисунок: {}", name)

    summary = _summary(points, sections, total, meta)
    title = f"Волновая активность береговых точек, {period}"
    (out_dir / "summary.md").write_text(
        f"# {title}\n\n" + "\n\n".join(f"## {head}\n\n" + "\n".join(f"- {line}" for line in lines) for head, lines in summary) + "\n",
        encoding="utf-8")
    sections_all.to_csv(out_dir / "sections.csv", index=False)
    gpd.GeoDataFrame(points[GEOJSON_COLUMNS], crs=4326, geometry=gpd.points_from_xy(points["lon"], points["lat"])).to_file(
        out_dir / "points_stats.geojson", driver="GeoJSON")
    _write_html(out_dir / "index.html", title, summary, figures, _interactive_map(points, max_html_points),
                _section_explorer(sections_all, monthly), sections_all)
    logger.success("Отчёт: {}", out_dir / "index.html")
    return {"html": out_dir / "index.html", "summary": out_dir / "summary.md", "geojson": out_dir / "points_stats.geojson",
            "sections": out_dir / "sections.csv", **{file.stem: file for file in figures.values()}}
