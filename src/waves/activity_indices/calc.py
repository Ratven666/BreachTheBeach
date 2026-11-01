from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    MIN_PERIODS_FOR_CV,
    MIN_PERIODS_FOR_TREND,
    ROUND_DIGITS,
    SIGN_EPSILON_WM,
)
from .periods import Period


@dataclass(frozen=True, slots=True)
class PeriodIndices:
    n_active_days: int
    normal_mean_wm: float
    alongshore_net_mean_wm: float
    alongshore_abs_mean_wm: float
    alongshore_pos_mean_wm: float
    alongshore_neg_mean_wm: float


def signed_angle_deg(wind_azimuth_deg: np.ndarray, normal_azimuth_deg: float) -> np.ndarray:
    """Знаковый угол wind - normal, приведённый к (-180, 180]."""
    delta = (np.asarray(wind_azimuth_deg, dtype=np.float64) - float(normal_azimuth_deg)) % 360.0
    return np.where(delta > 180.0, delta - 360.0, delta)


def alongshore_load_wm(
    cwef_wm: np.ndarray,
    wind_azimuth_deg: np.ndarray,
    normal_azimuth_deg: float,
    max_angle_deg: float,
) -> np.ndarray:
    """Вдольбереговая компонента P*sin(theta) = CWEF*tan(theta), Вт/м.

    Знак: theta > 0, если источник волн по часовой стрелке от нормали.
    Угол ограничивается max_angle_deg, чтобы tan не расходился у 90 градусов.
    """
    theta = signed_angle_deg(wind_azimuth_deg, normal_azimuth_deg)
    theta = np.clip(theta, -max_angle_deg, max_angle_deg)
    return np.asarray(cwef_wm, dtype=np.float64) * np.tan(np.radians(theta))


def aggregate_point(
    days: np.ndarray,
    cwef_wm: np.ndarray,
    alongshore_wm: np.ndarray,
    periods: list[Period],
) -> list[PeriodIndices]:
    """Средние за период по полной длине периода (нулевые дни учитываются)."""
    starts = np.array([p.start_day for p in periods], dtype=np.int64)
    n_periods = len(periods)
    idx = np.searchsorted(starts, days, side="right") - 1
    valid = (idx >= 0) & (idx < n_periods) & (days <= np.array([p.end_day for p in periods])[np.clip(idx, 0, n_periods - 1)])
    idx, c, a = idx[valid], cwef_wm[valid], alongshore_wm[valid]

    def sums(values: np.ndarray) -> np.ndarray:
        return np.bincount(idx, weights=values, minlength=n_periods)

    n_active = np.bincount(idx, minlength=n_periods)
    s_normal = sums(c)
    s_net = sums(a)
    s_abs = sums(np.abs(a))
    s_pos = sums(np.where(a > 0.0, a, 0.0))
    s_neg = sums(np.where(a < 0.0, -a, 0.0))
    length = np.array([p.n_days for p in periods], dtype=np.float64)

    return [
        PeriodIndices(
            n_active_days=int(n_active[i]),
            normal_mean_wm=float(s_normal[i] / length[i]),
            alongshore_net_mean_wm=float(s_net[i] / length[i]),
            alongshore_abs_mean_wm=float(s_abs[i] / length[i]),
            alongshore_pos_mean_wm=float(s_pos[i] / length[i]),
            alongshore_neg_mean_wm=float(s_neg[i] / length[i]),
        )
        for i in range(n_periods)
    ]


def _diff(cur: float, prev: float | None) -> tuple[float | None, float | None]:
    if prev is None:
        return None, None
    d = cur - prev
    pct = None if abs(prev) < SIGN_EPSILON_WM else 100.0 * d / abs(prev)
    return d, pct


def diffs_to_previous(items: list[PeriodIndices]) -> list[dict[str, float | None]]:
    out: list[dict[str, float | None]] = []
    for i, cur in enumerate(items):
        prev = items[i - 1] if i else None
        dn, dn_pct = _diff(cur.normal_mean_wm, prev.normal_mean_wm if prev else None)
        da, da_pct = _diff(cur.alongshore_abs_mean_wm, prev.alongshore_abs_mean_wm if prev else None)
        dnet, _ = _diff(cur.alongshore_net_mean_wm, prev.alongshore_net_mean_wm if prev else None)
        out.append(
            {
                "d_normal_wm": dn,
                "d_normal_pct": dn_pct,
                "d_alongshore_abs_wm": da,
                "d_alongshore_abs_pct": da_pct,
                "d_alongshore_net_wm": dnet,
            }
        )
    return out


def series_stats(years: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    """Среднее, CV между периодами, линейный тренд (OLS), R2, изменение первый->последний."""
    n = values.size
    mean = float(values.mean()) if n else None
    cv = None
    if n >= MIN_PERIODS_FOR_CV and mean is not None and abs(mean) > SIGN_EPSILON_WM:
        cv = float(values.std(ddof=1) / abs(mean))

    slope = r2 = None
    if n >= MIN_PERIODS_FOR_TREND and np.ptp(years) > 0.0:
        x = years - years.mean()
        sxx = float(np.sum(x * x))
        slope = float(np.sum(x * (values - mean)) / sxx)
        ss_tot = float(np.sum((values - mean) ** 2))
        ss_res = float(np.sum((values - mean - slope * x) ** 2))
        r2 = None if ss_tot <= SIGN_EPSILON_WM else float(1.0 - ss_res / ss_tot)

    first_last = None
    if n >= 2 and abs(values[0]) > SIGN_EPSILON_WM:
        first_last = float(100.0 * (values[-1] - values[0]) / abs(values[0]))

    return {"mean": mean, "cv": cv, "slope": slope, "r2": r2, "first_last_pct": first_last}


def sign_of(value: float) -> int:
    return 0 if abs(value) <= SIGN_EPSILON_WM else (1 if value > 0 else -1)


def r(value: float | None) -> float | None:
    return None if value is None else round(float(value), ROUND_DIGITS)
