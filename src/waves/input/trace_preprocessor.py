from __future__ import annotations

import pandas as pd

from src.waves.errors import WaveInputError


class TracePreprocessor:
    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower().replace(" ", "_"): c for c in df.columns}
        az = cols.get("direction") or cols.get("azimuth") or cols.get("azimuth_deg")
        dist = cols.get("fetch_m") or cols.get("distance_m") or cols.get("distance")

        if az is None or dist is None:
            raise WaveInputError(
                f"Trace input must contain direction and fetch_m columns. Got: {list(df.columns)}"
            )

        out = df[[az, dist]].copy()
        out.columns = ["direction", "fetch_m"]

        out["direction"] = (
            pd.to_numeric(out["direction"], errors="coerce")
            .round()
            .fillna(0)
            .astype(int)
            % 360
        )
        out["fetch_m"] = pd.to_numeric(out["fetch_m"], errors="coerce").fillna(0.0)

        out = (
            out.groupby("direction", as_index=False)["fetch_m"]
            .max()
            .sort_values("direction")
            .reset_index(drop=True)
        )
        return out
