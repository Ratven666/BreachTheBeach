from __future__ import annotations

import pandas as pd


def read_trace_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_wind_ts_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
