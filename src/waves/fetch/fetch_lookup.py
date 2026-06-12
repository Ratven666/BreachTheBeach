from __future__ import annotations

import numpy as np
import pandas as pd

from src.waves.errors import WaveInputError


class FetchLookup:
    def __init__(self, trace_df: pd.DataFrame) -> None:
        if trace_df.empty:
            raise WaveInputError("Trace dataframe is empty.")

        self._lookup = dict(
            zip(
                trace_df["direction"].astype(int),
                trace_df["fetch_m"].astype(float),
            )
        )
        self._directions = np.array(sorted(self._lookup.keys()), dtype=int)

    def get_fetch(self, direction_deg: int) -> float:
        direction_deg = int(direction_deg) % 360
        if direction_deg in self._lookup:
            return float(self._lookup[direction_deg])

        delta = np.minimum(
            np.abs(self._directions - direction_deg),
            360 - np.abs(self._directions - direction_deg),
        )
        idx = int(delta.argmin())
        nearest = int(self._directions[idx])
        return float(self._lookup[nearest])

    @property
    def directions(self) -> np.ndarray:
        return self._directions.copy()
