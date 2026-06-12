from __future__ import annotations

import numpy as np
import pandas as pd

from src.waves.errors import WaveInputError


class WindTimeSeriesPreprocessor:
    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        lower = {c.lower(): c for c in df.columns}

        c_date = lower.get("date")

        # Поддержка: уже нормализованный ws_kmh или оригинальные имена Open-Meteo
        c_ws = (
            lower.get("ws_kmh")
            or lower.get("wind_speed_10m_max")
            or lower.get("windspeed10mmax")
        )
        # Поддержка: уже нормализованный direction или оригинальные имена Open-Meteo
        c_dir = (
            lower.get("direction")
            or lower.get("wind_direction_10m_dominant")
            or lower.get("winddirection10mdominant")
            or lower.get("wind_dir")
        )

        if not all([c_date, c_ws, c_dir]):
            raise WaveInputError(
                "Wind time series must contain: date, "
                "wind_speed_10m_max (or ws_kmh), "
                f"wind_direction_10m_dominant (or direction). Got: {list(df.columns)}"
            )

        out = df[[c_date, c_ws, c_dir]].copy()
        out.columns = ["date", "ws_kmh", "direction"]

        out["date"] = pd.to_datetime(out["date"])
        out["ws_kmh"] = pd.to_numeric(out["ws_kmh"], errors="coerce").fillna(0.0)
        out["direction"] = (
            pd.to_numeric(out["direction"], errors="coerce").round().fillna(np.nan)
        )
        out = out.dropna(subset=["direction"]).copy()
        out["direction"] = out["direction"].astype(int) % 360

        return out.sort_values("date").reset_index(drop=True)
