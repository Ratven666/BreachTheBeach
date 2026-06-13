"""Сервис расчёта волнового воздействия для одной береговой точки.

Реализует цепочку:
    Wind (U, θ) → SMB (Hs_off, Tp) → Shoaling/Breaking (Green's law + Snell) → CWEF

Методика соответствует:
    - SMB: Shore Protection Manual (USACE, 1984)
    - Shoaling: Green's law  Hs ~ h^(-1/4)
    - Breaking: H_b = γ_b · h_b,  γ_b = 0.55
    - Рефракция: закон Снеллиуса cos(θ) через sqrt(g·h)
    - Мощность волны: P = ρg²Hs²Tp / (64π)
    - CWEF = P · cos(φ),  φ = angle(wave_dir − shore_normal)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

def _delta_deg(wave_dir: float, shore_normal: float) -> float:
    """Угол между направлением волны и нормалью к берегу, [-180, 180]."""
    d = wave_dir - shore_normal
    return float((d + 180) % 360 - 180)


def _cos_shore(delta_deg: float) -> float:
    """cos(δ), обрезанный снизу до 0 (волны, идущие от берега, игнорируются)."""
    return float(np.clip(np.cos(np.deg2rad(delta_deg)), 0.0, None))


# ──────────────────────────────────────────────────────────────────
# main class
# ──────────────────────────────────────────────────────────────────

@dataclass
class WaveClimateService:
    """Расчёт волнового климата для одной точки за временной ряд.

    Parameters
    ----------
    trace_df : DataFrame
        Таблица с колонками ``direction`` (int, 0–359) и ``fetch_m`` (float).
        Строки — азимуты трассировки разгона.
    wind_ts_df : DataFrame
        Временной ряд метео с колонками ``date``, ``wind_speed_10m_max`` (km/h),
        ``wind_direction_10m_dominant`` (°, 0–359).
    shore_normal_deg : float
        Азимут нормали к берегу, направленной в сторону моря.
    bathymetry_service : object | None
        Объект с методом ``get_profile(direction) -> profile`` для батиметрии.
        Если None — используются дефолтные глубины.
    origin_lon, origin_lat : float
        Координаты точки (для батиметрии).
    overwater_factor : float
        Поправочный коэффициент скорости ветра над водой (рекомендуется 1.1).
    breaking_coeff : float
        Коэффициент обрушения γ_b (обычно 0.55).
    default_h_deep_m : float
        Глубина в открытом море при отсутствии батиметрии.
    default_h_point_m : float
        Глубина у берега при отсутствии батиметрии.
    """

    trace_df: pd.DataFrame
    wind_ts_df: pd.DataFrame
    shore_normal_deg: float
    bathymetry_service: Optional[object] = None
    origin_lon: float = 0.0
    origin_lat: float = 0.0
    bathy_radius_m: float = 20_000.0
    bathy_n_steps: int = 200
    overwater_factor: float = 1.1
    breaking_coeff: float = 0.55
    rho_water: float = 1025.0
    g: float = 9.81
    default_h_deep_m: float = 20.0
    default_h_point_m: float = 3.0

    # внутренние — инициализируются в __post_init__
    _fetch_lookup: dict = field(default_factory=dict, init=False, repr=False)
    _profile_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_df = self._prepare_trace(self.trace_df.copy())
        self.wind_ts_df = self._prepare_wind_ts(self.wind_ts_df.copy())
        self._fetch_lookup = dict(
            zip(
                self.trace_df["direction"].astype(int),
                self.trace_df["fetch_m"].astype(float),
            )
        )

    # ── подготовка данных ──────────────────────────────────────────

    @staticmethod
    def _prepare_trace(df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация колонок трассировки; группировка по азимуту (берём max)."""
        cols = {c.lower().replace(" ", "_"): c for c in df.columns}
        az   = cols.get("direction") or cols.get("azimuth") or cols.get("azimuth_deg")
        dist = cols.get("fetch_m") or cols.get("fetch_length_m") or cols.get("distance_m")
        if az is None or dist is None:
            raise ValueError(
                f"trace_df должен иметь колонки direction и fetch_m, получено: {list(df.columns)}"
            )
        out = df[[az, dist]].copy()
        out.columns = ["direction", "fetch_m"]
        out["direction"] = pd.to_numeric(out["direction"], errors="coerce").round().fillna(0).astype(int) % 360
        out["fetch_m"]   = pd.to_numeric(out["fetch_m"],   errors="coerce").fillna(0.0)
        return (
            out.groupby("direction", as_index=False)["fetch_m"]
            .max()
            .sort_values("direction")
        )

    @staticmethod
    def _prepare_wind_ts(df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация колонок метеоряда; отбрасываем строки без направления."""
        lower = {c.lower(): c for c in df.columns}

        # ищем колонки под разными именами (weather GeoJSON vs CSV)
        c_date = (lower.get("date")
                  or lower.get("dates"))
        c_ws   = (lower.get("wind_speed_10m_max")
                  or lower.get("wind_speed")
                  or lower.get("ws_kmh"))
        c_dir  = (lower.get("wind_direction_10m_dominant")
                  or lower.get("wind_dir")
                  or lower.get("direction"))
        if not all([c_date, c_ws, c_dir]):
            raise ValueError(
                f"wind_ts_df должен иметь date, wind_speed и wind_dir, получено: {list(df.columns)}"
            )

        out = df[[c_date, c_ws, c_dir]].copy()
        out.columns = ["date", "ws_kmh", "direction"]
        out["date"]      = pd.to_datetime(out["date"])
        out["ws_kmh"]    = pd.to_numeric(out["ws_kmh"],    errors="coerce").fillna(0.0)
        out["direction"] = (
            pd.to_numeric(out["direction"], errors="coerce")
            .round()
            .fillna(np.nan)
        )
        # убираем строки без направления
        out = out.dropna(subset=["direction"]).copy()
        out["direction"] = out["direction"].astype(int) % 360
        return out.sort_values("date").reset_index(drop=True)

    # ── вспомогательные методы ─────────────────────────────────────

    def get_fetch(self, direction: int) -> float:
        """Возвращает длину разгона для азимута (ближайший в lookup)."""
        direction = int(direction) % 360
        if direction in self._fetch_lookup:
            return float(self._fetch_lookup[direction])
        dirs = np.array(list(self._fetch_lookup.keys()))
        # кольцевое расстояние
        diff = np.abs(dirs - direction)
        diff = np.minimum(diff, 360 - diff)
        idx  = int(diff.argmin())
        return float(self._fetch_lookup[dirs[idx]])

    def smb(self, U_ms: float, F_m: float) -> tuple[float, float]:
        """SMB-формулы (Shore Protection Manual, 1984).

        Parameters
        ----------
        U_ms : float  скорость ветра (м/с), с учётом overwater_factor
        F_m  : float  длина разгона (м)

        Returns
        -------
        Hs : float  значительная высота волны (м)
        Tp : float  пиковый период (с)
        """
        U = max(U_ms, 0.5)
        F = max(F_m,  1.0)
        X  = self.g * F / U**2
        Hs = 0.283 * (U**2 / self.g) * np.tanh(0.53 * X**0.75)
        Tp = 7.54  * (U   / self.g) * np.tanh(0.833 * X**0.375)
        return float(Hs), float(Tp)

    def _bathy_correction(
        self, direction: int, Hs_off: float, Tp: float
    ) -> tuple[float, float, float, float]:
        """Трансформация волны у берега.

        Возвращает (Hs_near, Ks, h_break, cos_refracted).

        Алгоритм:
            1. Shoaling по закону Грина:  Hs_shoaled = Hs_off * (h_deep/h_point)^0.25
            2. Breaking: ищем h_break в профиле где Hs_shoaled = γ_b * h
            3. Рефракция (Snell): sin(θ_out)/sqrt(g·h_point) = sin(θ_in)/sqrt(g·h_deep)
        """
        delta  = _delta_deg(direction, self.shore_normal_deg)
        cos_in = _cos_shore(delta)

        # ── без батиметрии ───────────────────────────────────────
        if self.bathymetry_service is None:
            h_deep  = self.default_h_deep_m
            h_point = self.default_h_point_m
            Ks       = (h_deep / max(h_point, 0.1)) ** 0.25
            Hs_shoal = Hs_off * Ks
            Hs_near  = min(Hs_shoal, self.breaking_coeff * h_point)
            # рефракция через дефолтные глубины
            C1 = np.sqrt(self.g * h_deep)
            C2 = np.sqrt(self.g * max(h_point, 0.1))
            sin_out = np.clip(np.sin(np.deg2rad(abs(delta))) * C2 / C1, -1.0, 1.0)
            cos_ref = float(np.clip(np.cos(np.arcsin(sin_out)), 0.0, None))
            return float(Hs_near), float(Ks), float(h_point), cos_ref

        # ── с батиметрией ────────────────────────────────────────
        if direction not in self._profile_cache:
            self._profile_cache[direction] = self.bathymetry_service.get_profile(direction)

        profile = self._profile_cache[direction]
        depths  = profile.depths_m  # массив глубин от точки к морю
        valid   = depths[np.isfinite(depths) & (depths > 0)]
        if valid.size == 0:
            # нет данных — дефолт
            h_deep  = self.default_h_deep_m
            h_point = self.default_h_point_m
        else:
            h_deep  = float(np.nanmax(valid))
            h_point = float(depths[0]) if (np.isfinite(depths[0]) and depths[0] > 0) else float(valid[0])

        Ks       = (h_deep / max(h_point, 0.1)) ** 0.25
        Hs_shoal = Hs_off * Ks

        # поиск глубины обрушения: идём от берега в море
        h_break = h_point
        for h in np.sort(valid):
            Hs_here = Hs_off * (h_deep / h) ** 0.25
            if Hs_here >= self.breaking_coeff * h:
                h_break = float(h)
                break

        Hs_near = min(Hs_shoal, self.breaking_coeff * h_break)

        # рефракция Снеллиус
        C1 = np.sqrt(self.g * h_deep)
        C2 = np.sqrt(self.g * max(h_point, 0.1))
        sin_out = np.clip(np.sin(np.deg2rad(abs(delta))) * C2 / C1, -1.0, 1.0)
        cos_ref = float(np.clip(np.cos(np.arcsin(sin_out)), 0.0, None))

        return float(Hs_near), float(Ks), float(h_break), cos_ref

    # ── основные расчёты ──────────────────────────────────────────

    def calculate_daily(self) -> pd.DataFrame:
        """Рассчитывает дневной ряд параметров волнового воздействия.

        Возвращаемые колонки
        --------------------
        date, direction, fetch_m, U10_ms,
        Hs_offshore_m, Hs_nearshore_m, Tp_s,
        Ks, h_breaking_m, cos_shore,
        WavePower_W_m, CWEF_W_m
        """
        records = []
        for _, row in self.wind_ts_df.iterrows():
            U_kmh = float(row["ws_kmh"])
            if U_kmh < 0.1:
                continue

            U_ms      = U_kmh / 3.6 * self.overwater_factor
            direction = int(row["direction"])
            F         = self.get_fetch(direction)

            Hs_off, Tp = self.smb(U_ms, F)

            Hs_near, Ks, h_break, cos_ref = self._bathy_correction(
                direction, Hs_off, Tp
            )

            # Мощность волны: P = ρg²Hs²Tp / (64π)
            P    = self.rho_water * self.g**2 * Hs_near**2 * Tp / (64.0 * np.pi)
            CWEF = P * cos_ref

            records.append({
                "date":             row["date"],
                "direction":        direction,
                "fetch_m":          round(F, 1),
                "U10_ms":           round(U_ms, 3),
                "Hs_offshore_m":    round(Hs_off,  4),
                "Hs_nearshore_m":   round(Hs_near, 4),
                "Tp_s":             round(Tp,      4),
                "Ks":               round(Ks,      4),
                "h_breaking_m":     round(h_break, 3),
                "cos_shore":        round(cos_ref,  4),
                "WavePower_W_m":    round(P,        3),
                "CWEF_W_m":         round(CWEF,     3),
            })

        return pd.DataFrame(records)

    def cwef_stats(self, daily: pd.DataFrame | None = None) -> dict:
        """Агрегированная статистика CWEF за весь период.

        Parameters
        ----------
        daily : DataFrame | None
            Результат calculate_daily(); если None — вычисляется автоматически.
        """
        if daily is None:
            daily = self.calculate_daily()
        c = daily["CWEF_W_m"]
        n = len(c)
        p90 = float(c.quantile(0.90))
        return {
            "shore_normal_deg":  round(self.shore_normal_deg, 1),
            "n_days_total":      n,
            "n_days_active":     int((c > 0).sum()),
            "mean_W_m":          round(float(c.mean()),             2),
            "median_W_m":        round(float(c.median()),           2),
            "std_W_m":           round(float(c.std()),              2),
            "p75_W_m":           round(float(c.quantile(0.75)),     2),
            "p90_W_m":           round(p90,                         2),
            "p95_W_m":           round(float(c.quantile(0.95)),     2),
            "p99_W_m":           round(float(c.quantile(0.99)),     2),
            "max_W_m":           round(float(c.max()),              2),
            "n_storm_days_p90":  int((c >= p90).sum()),
            "total_energy_MJ_m": round(float(c.sum()) * 86400 / 1e6, 1),
            # доп. статистика по высоте волны
            "Hs_mean_m":         round(float(daily["Hs_nearshore_m"].mean()), 3),
            "Hs_max_m":          round(float(daily["Hs_nearshore_m"].max()),  3),
        }
