from __future__ import annotations

"""Векторные статистические оценки. Все ряды: массив [n_рядов, n_лет]."""

import numpy as np
from scipy.stats import false_discovery_control, norm, rankdata


def mann_kendall(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Наклон Тейла–Сена (ед./год), Z и p-value теста Манна–Кендалла с поправкой на связки."""
    n = x.shape[1]
    i, j = np.triu_indices(n, 1)
    diff = x[:, j] - x[:, i]
    slope = np.median(diff / (t[j] - t[i]), axis=1)
    s = np.sign(diff).sum(axis=1)
    ties = (x[:, :, None] == x[:, None, :]).sum(axis=2)          # размер группы равных значений
    var = (n * (n - 1) * (2 * n + 5) - ((ties - 1) * (2 * ties + 5)).sum(axis=1)) / 18.0
    z = np.where(var > 0, (s - np.sign(s)) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    return slope, z, 2.0 * norm.sf(np.abs(z))


def trend(x: np.ndarray, t: np.ndarray, prewhiten: bool = True):
    """Тренд ряда: наклон Сена, Z, p. prewhiten — TFPW (Yue et al., 2002).

    Если автокорреляция остатков r1 значима, тест выполняется по ряду
    y_t = (x_t - b*t) - r1*(x_{t-1} - b*(t-1)) + b*t.
    """
    t = np.asarray(t, dtype=float)
    slope, z, p = mann_kendall(x, t)
    if not prewhiten or x.shape[1] < 10:
        return slope, z, p
    resid = x - slope[:, None] * (t - t[0])
    dev = resid - resid.mean(axis=1, keepdims=True)
    ss = (dev**2).sum(axis=1)
    r1 = np.where(ss > 0, (dev[:, 1:] * dev[:, :-1]).sum(axis=1) / np.maximum(ss, 1e-12), 0.0)
    use = np.abs(r1) > 1.96 / np.sqrt(x.shape[1])
    y = resid[:, 1:] - r1[:, None] * resid[:, :-1] + slope[:, None] * (t[1:] - t[0])
    _, z_pw, p_pw = mann_kendall(y, t[1:])
    return slope, np.where(use, z_pw, z), np.where(use, p_pw, p)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """q-value Бенджамини–Хохберга (контроль доли ложных открытий по всем точкам)."""
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = false_discovery_control(p[ok])
    return q


def classify(slope, pct_dec, q, alpha: float, min_pct: float, strong_pct: float) -> np.ndarray:
    """Класс изменения: -2..2. Значимо, если q <= alpha и |тренд| >= min_pct % за 10 лет."""
    mag = np.abs(np.nan_to_num(pct_dec))
    significant = (np.nan_to_num(q, nan=1.0) <= alpha) & (mag >= min_pct)
    return (np.sign(slope) * significant * (1 + (mag >= strong_pct))).astype(int)


def pettitt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Тест Петтитта: индекс первого года после сдвига и приближённое p-value."""
    n = x.shape[1]
    u = np.abs(2.0 * np.cumsum(rankdata(x, axis=1), axis=1) - np.arange(1, n + 1) * (n + 1))[:, :-1]
    k = u.argmax(axis=1)
    p = np.minimum(1.0, 2.0 * np.exp(-6.0 * u.max(axis=1) ** 2 / (n**3 + n**2)))
    return k + 1, p


def gumbel_levels(annual_max: np.ndarray, periods) -> dict[int, np.ndarray]:
    """Уровни повторяемости (раз в T лет) по годовым максимумам, Гумбель, метод моментов."""
    beta = annual_max.std(axis=1, ddof=1) * np.sqrt(6.0) / np.pi
    mu = annual_max.mean(axis=1) - 0.5772 * beta
    return {int(T): mu - beta * np.log(-np.log(1.0 - 1.0 / T)) for T in periods}


def seasonal_concentration(share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Индекс сезонной концентрации (0 — равномерно, 1 — всё в одном месяце)
    и «центр тяжести» года в месяцах (1.0 — середина января). share: [n, 12], сумма = 1."""
    angle = 2.0 * np.pi * (np.arange(12) + 0.5) / 12.0
    c, s = share @ np.cos(angle), share @ np.sin(angle)
    return np.hypot(c, s), (np.degrees(np.arctan2(s, c)) % 360.0) / 30.0 + 0.5


def percentile_with_zeros(active_sorted: np.ndarray, n_total: int, q) -> np.ndarray:
    """Перцентили ряда из n_total дней, где хранятся только ненулевые значения."""
    if active_sorted.size == 0:
        return np.zeros(len(q))
    h = (n_total - 1) * np.asarray(q, dtype=float) / 100.0
    lo = np.floor(h).astype(int)
    n_zero = n_total - active_sorted.size

    def value(k):
        return np.where(k >= n_zero, active_sorted[np.clip(k - n_zero, 0, None)], 0.0)

    return value(lo) + (h - lo) * (value(np.minimum(lo + 1, n_total - 1)) - value(lo))


def wrap180(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0
