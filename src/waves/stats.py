from __future__ import annotations

import pandas as pd


class WaveClimateStatistics:
    SECTOR_LABELS = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]

    @staticmethod
    def cwef_stats(daily: pd.DataFrame, shore_normal_deg: float) -> dict:
        """Агрегированная статистика CWEF за весь период.

        Если daily пуст (все дни без ветра или точка пропущена) —
        возвращает словарь с None-значениями вместо ValueError
        из round(float(NaN)).
        """
        base = {"shore_normal_deg": round(float(shore_normal_deg), 1)}

        if daily.empty or "CWEF_Wm" not in daily.columns:
            return {
                **base,
                "mean_Wm": None,
                "median_Wm": None,
                "std_Wm": None,
                "p75_Wm": None,
                "p90_Wm": None,
                "p95_Wm": None,
                "p99_Wm": None,
                "max_Wm": None,
                "n_days": 0,
                "n_storm_days_p90": 0,
                "total_energy_MJm": 0.0,
            }

        c = daily["CWEF_Wm"]
        n = len(c)
        p90 = c.quantile(0.90)
        return {
            **base,
            "mean_Wm": round(float(c.mean()), 2),
            "median_Wm": round(float(c.median()), 2),
            "std_Wm": round(float(c.std()), 2),
            "p75_Wm": round(float(c.quantile(0.75)), 2),
            "p90_Wm": round(float(p90), 2),
            "p95_Wm": round(float(c.quantile(0.95)), 2),
            "p99_Wm": round(float(c.quantile(0.99)), 2),
            "max_Wm": round(float(c.max()), 2),
            "n_days": int(n),
            "n_storm_days_p90": int((c >= p90).sum()),
            "total_energy_MJm": round(float(c.sum() * 86400.0 / 1e6), 1),
        }

    @staticmethod
    def cwef_by_direction(daily: pd.DataFrame) -> pd.DataFrame:
        return (
            daily.groupby("direction", as_index=False)
            .agg(
                mean_CWEF_Wm=("CWEF_Wm", "mean"),
                sum_CWEF_Wm=("CWEF_Wm", "sum"),
                mean_Hs_near=("Hs_nearshore_m", "mean"),
                max_Hs_near=("Hs_nearshore_m", "max"),
                n_days=("CWEF_Wm", "count"),
            )
            .sort_values("sum_CWEF_Wm", ascending=False)
            .reset_index(drop=True)
        )

    @classmethod
    def cwef_by_sector(cls, daily: pd.DataFrame) -> pd.DataFrame:
        df = daily.copy()
        df["sector"] = [
            cls.SECTOR_LABELS[int((d + 11.25) / 22.5) % 16]
            for d in df["direction"]
        ]
        order = {lab: i for i, lab in enumerate(cls.SECTOR_LABELS)}
        out = (
            df.groupby("sector", as_index=False)
            .agg(
                sum_CWEF_Wm=("CWEF_Wm", "sum"),
                mean_CWEF_Wm=("CWEF_Wm", "mean"),
                mean_Hs_near=("Hs_nearshore_m", "mean"),
                max_Hs_near=("Hs_nearshore_m", "max"),
                mean_cos_shore=("cos_shore", "mean"),
                n_days=("CWEF_Wm", "count"),
            )
        )
        out["o"] = out["sector"].map(order)
        return out.sort_values("o").drop(columns="o").reset_index(drop=True)

    @staticmethod
    def annual_extremes(daily: pd.DataFrame) -> pd.DataFrame:
        df = daily.copy()
        # Страховка: date может прийти строкой после десериализации из GeoJSON
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        return df.groupby("year", as_index=False).agg(
            Hs_max_m=("Hs_nearshore_m", "max"),
            CWEF_max_Wm=("CWEF_Wm", "max"),
            mean_CWEF_Wm=("CWEF_Wm", "mean"),
        )
