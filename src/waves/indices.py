"""Модуль расчёта индекса волновой экспозиции (Wave Exposure Rank, WER).

Реализует четыре частных показателя и итоговый ранг:

    WER = (R1 · R2 · R3 · R4)^(1/4)          (1)

Частные ранги R1–R4 ∈ [1, 5] присваиваются квинтильным методом
(или явными порогами) по ансамблю всех точек трассы:

    R1 — ранг по среднегодовому CWEF̄  [Вт/м]   (фоновая нагрузка)
    R2 — ранг по E_storm              [МДж/м]  (экстремальная составляющая)
    R3 — ранг по K_dir                [0..1]   (направленная концентрация)
    R4 — ранг по CV = σ/CWEF̄          [-]      (межгодовая изменчивость)

Примечание по именам колонок
-----------------------------
WaveClimateService.calculate_daily() пишет колонку «CWEF_W_m».
stats.py ожидает «CWEF_Wm».
Модуль нормализует оба варианта через _normalize_columns().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_LABELS: list[str] = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

_DT_SECONDS: float = 86_400.0

# Все возможные имена колонки CWEF
_CWEF_ALIASES: tuple[str, ...] = (
    "CWEF_Wm", "CWEF_W_m", "cwef_wm", "cwef_w_m", "CWEF"
)
_DIR_ALIASES: tuple[str, ...] = (
    "direction", "Direction", "dir_deg", "wave_dir"
)

# Квинтильные границы — 5 равных частей [1, 5]
_QUINTILE_BOUNDS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Нормализация входного DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_columns(daily: pd.DataFrame) -> pd.DataFrame:
    """Возвращает копию daily с унифицированными именами колонок.

    Переименовывает любой вариант имени CWEF → «CWEF_Wm».
    """
    rename: dict[str, str] = {}
    cols = set(daily.columns)

    if "CWEF_Wm" not in cols:
        for alias in _CWEF_ALIASES:
            if alias in cols:
                rename[alias] = "CWEF_Wm"
                break

    if "direction" not in cols:
        for alias in _DIR_ALIASES:
            if alias in cols:
                rename[alias] = "direction"
                break

    return daily.rename(columns=rename) if rename else daily.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _direction_to_sector(direction_deg: float) -> str:
    """Переводит азимут (0–360) в 16-румбовый сектор."""
    idx = int((float(direction_deg) + 11.25) / 22.5) % 16
    return SECTOR_LABELS[idx]


def _top3_adjacent_sectors(sector_sums: dict[str, float]) -> list[str]:
    """Три *соседних* румба с наибольшей суммой CWEF (скользящее окно по кольцу)."""
    n = len(SECTOR_LABELS)
    ring = SECTOR_LABELS * 2
    best_sum = -1.0
    best_idx = 0
    for i in range(n):
        s = sum(sector_sums.get(lab, 0.0) for lab in ring[i: i + 3])
        if s > best_sum:
            best_sum = s
            best_idx = i
    return ring[best_idx: best_idx + 3]


def assign_quintile_ranks(series: pd.Series) -> pd.Series:
    """Присваивает ранг 1–5 по квинтилям ансамбля (без учёта NaN).

    Все значения ниже 20-го перцентиля → ранг 1 (минимальная экспозиция),
    выше 80-го перцентиля → ранг 5 (максимальная).

    Parameters
    ----------
    series : pd.Series
        Вектор показателей по всем точкам ансамбля.

    Returns
    -------
    pd.Series
        Целочисленные ранги 1–5 той же длины и с тем же индексом.
        NaN-значения в исходном ряду получают NaN в результате.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    # Вычисляем квинтильные пороги по непустым значениям
    thresholds = [float(valid.quantile(q)) for q in _QUINTILE_BOUNDS[1:-1]]
    # thresholds = [p20, p40, p60, p80]

    def _rank_one(v: float) -> int:
        if v <= thresholds[0]:
            return 1
        if v <= thresholds[1]:
            return 2
        if v <= thresholds[2]:
            return 3
        if v <= thresholds[3]:
            return 4
        return 5

    return series.apply(lambda x: _rank_one(x) if pd.notna(x) else np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс — расчёт показателей для одной точки
# ─────────────────────────────────────────────────────────────────────────────

class WaveExposureIndex:
    """Рассчитывает четыре частных показателя WEI из дневного ряда CWEF.

    Ранги R1–R4 и итоговый WER рассчитываются **ансамблево** через
    :class:`WaveExposureRanker` после агрегации показателей по всем точкам.

    Parameters
    ----------
    storm_percentile : float
        Перцентиль для штормового порога (по умолчанию 90).
    storm_threshold : float | None
        Явно заданный порог [Вт/м]. Если задан — percentile игнорируется.
    dt_seconds : float
        Временной шаг ряда в секундах (86 400 для суточных данных).
    """

    def __init__(
        self,
        storm_percentile: float = 90.0,
        storm_threshold: float | None = None,
        dt_seconds: float = _DT_SECONDS,
    ) -> None:
        if not (0 < storm_percentile < 100):
            raise ValueError("storm_percentile должен быть в диапазоне (0, 100)")
        self.storm_percentile = storm_percentile
        self.storm_threshold = storm_threshold
        self.dt_seconds = dt_seconds

    # ── публичный API ─────────────────────────────────────────────────────────

    def compute(self, daily: pd.DataFrame) -> dict:
        """Вычисляет четыре частных показателя для одной точки.

        Parameters
        ----------
        daily : pd.DataFrame
            Дневной ряд одной точки. Принимает CWEF как «CWEF_Wm» или
            «CWEF_W_m» (нормализуется автоматически).

        Returns
        -------
        dict
            Ключи: ``mean_CWEF_Wm``, ``E_storm_MJm``, ``storm_threshold_Wm``,
            ``K_dir``, ``CV``, ``n_days``, ``n_storm_days``,
            ``top3_sectors``, ``storm_percentile``.
            Ранги R1–R4 и WER здесь **не вычисляются** — они требуют
            ансамбля всех точек и рассчитываются в :class:`WaveExposureRanker`.
        """
        if daily.empty:
            return self._empty_result()

        df = _normalize_columns(daily)
        if "CWEF_Wm" not in df.columns:
            return self._empty_result()

        cwef = df["CWEF_Wm"].astype(float)
        n = len(cwef)

        # 1. Среднее CWEF (основа R1)
        mean_cwef = float(cwef.mean())

        # 2. Штормовая энергия (основа R2)
        thresh = (
            self.storm_threshold
            if self.storm_threshold is not None
            else float(cwef.quantile(self.storm_percentile / 100.0))
        )
        storm_mask = cwef > thresh
        e_storm_mjm = float((cwef[storm_mask] * self.dt_seconds).sum() / 1e6)
        n_storm = int(storm_mask.sum())

        # 3. Коэффициент направленной концентрации (основа R3)
        k_dir, top3 = self._compute_k_dir(df, cwef)

        # 4. Коэффициент вариации (основа R4)
        cv = self._compute_cv(cwef, mean_cwef)

        return {
            "mean_CWEF_Wm":       round(mean_cwef, 3),
            "E_storm_MJm":        round(e_storm_mjm, 3),
            "storm_threshold_Wm": round(thresh, 3),
            "storm_percentile":   self.storm_percentile,
            "K_dir":              round(k_dir, 4),
            "top3_sectors":       ",".join(top3),
            "CV":                 round(cv, 4),
            "n_days":             n,
            "n_storm_days":       n_storm,
        }

    def compute_annual(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет показатели по годам (межгодовая динамика).

        Returns
        -------
        pd.DataFrame
            Строка на год: year + все ключи из compute().
        """
        df = _normalize_columns(daily).copy()
        if "year" not in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year

        records = []
        for year, grp in df.groupby("year"):
            rec = self.compute(grp.reset_index(drop=True))
            rec["year"] = int(year)
            records.append(rec)

        if not records:
            return pd.DataFrame()

        out = pd.DataFrame(records)
        cols = ["year", "mean_CWEF_Wm", "E_storm_MJm",
                "storm_threshold_Wm", "K_dir", "CV",
                "n_days", "n_storm_days", "top3_sectors"]
        cols = [c for c in cols if c in out.columns]
        return out[cols].sort_values("year").reset_index(drop=True)

    # ── вспомогательные методы ────────────────────────────────────────────────

    def _compute_k_dir(
        self, df: pd.DataFrame, cwef: pd.Series
    ) -> tuple[float, list[str]]:
        if "direction" not in df.columns:
            return 0.0, []
        tmp = df[["direction", "CWEF_Wm"]].copy()
        tmp["_sector"] = tmp["direction"].apply(_direction_to_sector)
        sector_sums: dict[str, float] = (
            tmp.groupby("_sector")["CWEF_Wm"].sum().to_dict()
        )
        total_sum = sum(sector_sums.values())
        if total_sum == 0:
            return 0.0, []
        top3 = _top3_adjacent_sectors(sector_sums)
        k_dir = float(sum(sector_sums.get(lab, 0.0) for lab in top3) / total_sum)
        return k_dir, top3

    @staticmethod
    def _compute_cv(cwef: pd.Series, mean_cwef: float) -> float:
        if mean_cwef == 0 or np.isnan(mean_cwef):
            return 0.0
        return float(cwef.std(ddof=1) / mean_cwef)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "mean_CWEF_Wm":       None,
            "E_storm_MJm":        None,
            "storm_threshold_Wm": None,
            "storm_percentile":   None,
            "K_dir":              None,
            "top3_sectors":       None,
            "CV":                 None,
            "n_days":             0,
            "n_storm_days":       0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ансамблевое ранжирование — WER по формуле (1)
# ─────────────────────────────────────────────────────────────────────────────

class WaveExposureRanker:
    """Присваивает ранги R1–R4 и вычисляет WER по ансамблю точек.

    WER = (R1 · R2 · R3 · R4)^(1/4)                              (1)

    R1 — ранг по mean_CWEF_Wm   (фоновая нагрузка)
    R2 — ранг по E_storm_MJm    (экстремальная составляющая)
    R3 — ранг по K_dir          (направленная концентрация)
    R4 — ранг по CV             (межгодовая изменчивость)

    Каждый ранг принимает целочисленные значения 1–5 по квинтилям
    распределения показателя по всем точкам ансамбля.

    Parameters
    ----------
    metric_cols : dict[str, str] | None
        Маппинг «имя показателя» → «имя колонки в DataFrame».
        По умолчанию используется стандартный маппинг:
        ``{"R1": "mean_CWEF_Wm", "R2": "E_storm_MJm",
           "R3": "K_dir", "R4": "CV"}``.
    """

    _DEFAULT_METRICS: dict[str, str] = {
        "R1": "mean_CWEF_Wm",
        "R2": "E_storm_MJm",
        "R3": "K_dir",
        "R4": "CV",
    }

    def __init__(
        self,
        metric_cols: dict[str, str] | None = None,
    ) -> None:
        self.metric_cols: dict[str, str] = (
            metric_cols if metric_cols is not None else dict(self._DEFAULT_METRICS)
        )

    def rank(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет колонки R1, R2, R3, R4 и WER к сводному DataFrame.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Таблица с одной строкой на точку, содержащая колонки
            ``mean_CWEF_Wm``, ``E_storm_MJm``, ``K_dir``, ``CV``.

        Returns
        -------
        pd.DataFrame
            Копия ``summary_df`` с добавленными колонками:
            ``R1``, ``R2``, ``R3``, ``R4``, ``WER``.
        """
        out = summary_df.copy()

        for rank_col, metric_col in self.metric_cols.items():
            if metric_col not in out.columns:
                out[rank_col] = np.nan
                continue
            out[rank_col] = assign_quintile_ranks(out[metric_col]).astype("Int64")

        # WER = геометрическое среднее R1..R4
        rank_cols = list(self.metric_cols.keys())
        r_arr = out[rank_cols].astype(float).values  # shape (n_points, 4)

        # Строки где хотя бы один ранг NaN → WER = NaN
        has_nan = np.any(np.isnan(r_arr), axis=1)
        product = np.where(has_nan, np.nan, np.prod(r_arr, axis=1))
        wer = np.where(np.isnan(product), np.nan, product ** (1.0 / len(rank_cols)))

        out["WER"] = np.round(wer, 3)
        return out

    def rank_gdf(self, gdf):
        """Обёртка для GeoDataFrame — сохраняет геометрию и CRS."""
        import geopandas as gpd
        out = self.rank(pd.DataFrame(gdf.drop(columns=["geometry"])))
        out["geometry"] = gdf["geometry"].values
        return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)
