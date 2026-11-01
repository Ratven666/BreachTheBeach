from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NearshoreWaveRecord:
    hs_nearshore_m: float
    ks: float
    h_breaking_m: float
    cos_shore: float
    refracted_angle_deg: float | None = None
