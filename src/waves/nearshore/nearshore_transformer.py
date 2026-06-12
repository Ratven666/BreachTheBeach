from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.waves.domain import NearshoreWaveRecord
from src.waves.errors import WaveBathymetryError
from src.waves.nearshore.bathymetry_adapter import BathymetryProfileProvider
from src.waves.nearshore.breaking import BreakingModel
from src.waves.nearshore.refraction import RefractionModel
from src.waves.nearshore.shoaling import ShoalingModel


@dataclass
class NearshoreWaveTransformer:
    shore_normal_deg: float
    profile_provider: BathymetryProfileProvider | None = None
    shoaling_model: ShoalingModel = field(default_factory=ShoalingModel)
    breaking_model: BreakingModel = field(default_factory=BreakingModel)
    refraction_model: RefractionModel = field(default_factory=RefractionModel)
    _profile_cache: dict[int, object] = field(default_factory=dict, init=False, repr=False)

    def transform(self, direction_deg: int, hs_offshore: float, tp_s: float) -> NearshoreWaveRecord:
        if self.profile_provider is None:
            h = 10.0
            hs_near = min(float(hs_offshore), self.breaking_model.gamma_b * h)
            cos_shore, theta_out_deg = self.refraction_model.transform(
                direction_deg=float(direction_deg),
                shore_normal_deg=self.shore_normal_deg,
                h_deep=h,
                h_point=h,
            )
            return NearshoreWaveRecord(
                hs_nearshore_m=float(hs_near),
                ks=1.0,
                h_breaking_m=float(h),
                cos_shore=float(cos_shore),
                refracted_angle_deg=float(theta_out_deg),
            )

        if direction_deg not in self._profile_cache:
            self._profile_cache[direction_deg] = self.profile_provider.get_profile(direction_deg)

        profile = self._profile_cache[direction_deg]
        depths = np.asarray(profile.depths_m, dtype=float)
        valid = depths[np.isfinite(depths) & (depths > 0)]
        if valid.size == 0:
            raise WaveBathymetryError(
                f"No positive valid bathymetric depths found for direction {direction_deg}°."
            )

        h_deep = float(np.nanmax(valid))
        h_point = float(depths[0]) if (np.isfinite(depths[0]) and depths[0] > 0) else float(valid[0])
        hs_shoaled, ks = self.shoaling_model.transform(hs_offshore, h_deep, h_point)
        hs_break, h_break = self.breaking_model.apply(hs_offshore, h_deep, valid)
        hs_near = min(float(hs_shoaled), float(hs_break))
        cos_shore, theta_out_deg = self.refraction_model.transform(
            direction_deg=float(direction_deg),
            shore_normal_deg=self.shore_normal_deg,
            h_deep=h_deep,
            h_point=h_point,
        )
        return NearshoreWaveRecord(
            hs_nearshore_m=float(hs_near),
            ks=float(ks),
            h_breaking_m=float(h_break),
            cos_shore=float(cos_shore),
            refracted_angle_deg=float(theta_out_deg),
        )
