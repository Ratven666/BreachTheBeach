from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FetchRecord:
    direction_deg: int
    fetch_m: float


@dataclass(frozen=True, slots=True)
class WindRecord:
    date: pd.Timestamp
    speed_10m_kmh: float
    direction_deg: int


@dataclass(frozen=True, slots=True)
class OffshoreWaveRecord:
    hs_m: float
    tp_s: float
    fetch_m: float
    u10_ms: float


@dataclass(frozen=True, slots=True)
class NearshoreWaveRecord:
    hs_nearshore_m: float
    ks: float
    h_breaking_m: float
    cos_shore: float
    refracted_angle_deg: float | None = None


@dataclass(frozen=True, slots=True)
class DailyWaveClimateRecord:
    date: pd.Timestamp
    direction_deg: int
    fetch_m: float
    u10_ms: float
    hs_offshore_m: float
    hs_nearshore_m: float
    tp_s: float
    ks: float
    h_breaking_m: float
    wave_power_wm: float
    cos_shore: float
    cwef_wm: float


@dataclass(frozen=True, slots=True)
class WaveClimateSummary:
    shore_normal_deg: float
    mean_wm: float
    median_wm: float
    std_wm: float
    p75_wm: float
    p90_wm: float
    p95_wm: float
    p99_wm: float
    max_wm: float
    n_days: int
    n_storm_days_p90: int
    total_energy_mjm: float
