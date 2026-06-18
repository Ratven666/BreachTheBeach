from __future__ import annotations

import numpy as np
import pandas as pd

from src.waves.errors import WaveInputError


class ShoreNormalEstimator:
    @staticmethod
    def estimate(trace_df: pd.DataFrame, window: int = 11) -> float:
        if trace_df.empty:
            raise WaveInputError("Trace dataframe is empty; cannot estimate shore normal.")

        df = trace_df.sort_values("direction")
        fetch = df["fetch_m"].to_numpy(dtype=float)
        directions = df["direction"].to_numpy(dtype=float)
        smooth = pd.Series(fetch).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
        idx = int(np.argmax(smooth))
        return float(directions[idx])
