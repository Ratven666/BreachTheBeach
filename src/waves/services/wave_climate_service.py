from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.waves.energy import WaveEnergyCalculator
from src.waves.fetch import FetchLookup
from src.waves.input import TracePreprocessor, WindTimeSeriesPreprocessor, read_trace_csv, read_wind_ts_csv
from src.waves.nearshore import BathymetryProfileProvider, NearshoreWaveTransformer, BreakingModel, RefractionModel, ShoalingModel
from src.waves.offshore import SMBWaveGrowthModel
from src.waves.shoreline import ShoreNormalEstimator
from src.waves.stats import WaveClimateStatistics


@dataclass
class WaveClimateService:
    trace_df: pd.DataFrame
    wind_ts_df: pd.DataFrame
    shore_normal_deg: Optional[float] = None
    bathymetry_service: object | None = None
    origin_lon: float = 0.0
    origin_lat: float = 0.0
    bathy_radius_m: float = 20_000.0
    bathy_n_steps: int = 200
    breaking_coeff: float = 0.55
    overwater_factor: float = 1.1
    rho_water: float = 1025.0
    g: float = 9.81

    def __post_init__(self) -> None:
        self.trace_df = TracePreprocessor.prepare(self.trace_df.copy())
        self.wind_ts_df = WindTimeSeriesPreprocessor.prepare(self.wind_ts_df.copy())
        self._fetch_lookup = FetchLookup(self.trace_df)
        self._shore_normal = float(self.shore_normal_deg) if self.shore_normal_deg is not None else ShoreNormalEstimator.estimate(self.trace_df)
        self._offshore_model = SMBWaveGrowthModel(g=self.g)
        self._energy = WaveEnergyCalculator(rho_water=self.rho_water, g=self.g)
        profile_provider = None
        if self.bathymetry_service is not None:
            profile_provider = BathymetryProfileProvider(
                bathymetry_service=self.bathymetry_service,
                origin_lon=self.origin_lon,
                origin_lat=self.origin_lat,
                radius_m=self.bathy_radius_m,
                n_steps=self.bathy_n_steps,
            )
        self._nearshore = NearshoreWaveTransformer(
            shore_normal_deg=self._shore_normal,
            profile_provider=profile_provider,
            shoaling_model=ShoalingModel(),
            breaking_model=BreakingModel(gamma_b=self.breaking_coeff),
            refraction_model=RefractionModel(g=self.g),
        )

    @classmethod
    def from_csv(cls, trace_csv: str, wind_ts_csv: str, **kwargs) -> "WaveClimateService":
        return cls(trace_df=read_trace_csv(trace_csv), wind_ts_df=read_wind_ts_csv(wind_ts_csv), **kwargs)

    @property
    def shore_normal(self) -> float:
        return float(self._shore_normal)

    def calculate_daily(self) -> pd.DataFrame:
        records: list[dict] = []
        for _, row in self.wind_ts_df.iterrows():
            u_kmh = float(row["ws_kmh"])
            if u_kmh <= 0.1:
                continue
            u_ms = u_kmh / 3.6 * self.overwater_factor
            direction = int(row["direction"])
            fetch_m = self._fetch_lookup.get_fetch(direction)
            hs_off, tp = self._offshore_model.calculate(u_ms, fetch_m)
            near = self._nearshore.transform(direction_deg=direction, hs_offshore=hs_off, tp_s=tp)
            power = self._energy.wave_power(near.hs_nearshore_m, tp)
            cwef = self._energy.cwef(power, near.cos_shore)
            records.append(
                {
                    "date": row["date"],
                    "direction": direction,
                    "fetch_m": round(fetch_m, 3),
                    "U10_ms": round(u_ms, 3),
                    "Hs_offshore_m": round(hs_off, 4),
                    "Hs_nearshore_m": round(near.hs_nearshore_m, 4),
                    "Tp_s": round(tp, 4),
                    "Ks": round(near.ks, 4),
                    "h_breaking_m": round(near.h_breaking_m, 3),
                    "refracted_angle_deg": round(float(near.refracted_angle_deg or 0.0), 3),
                    "WavePower_Wm": round(power, 3),
                    "cos_shore": round(near.cos_shore, 4),
                    "CWEF_Wm": round(cwef, 3),
                }
            )
        return pd.DataFrame(records)

    def cwef_stats(self, daily: Optional[pd.DataFrame] = None) -> dict:
        if daily is None:
            daily = self.calculate_daily()
        return WaveClimateStatistics.cwef_stats(daily, self._shore_normal)
