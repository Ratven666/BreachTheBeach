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
    default_h_deep_m: float = 20.0
    default_h_point_m: float = 3.0
    _profile_cache: dict[int, object] = field(default_factory=dict, init=False, repr=False)

    def transform(
        self,
        direction_deg: int,
        hs_offshore: float,
        tp_s: float,
    ) -> NearshoreWaveRecord:

        # ── Режим без батиметрии ─────────────────────────────────────────────
        if self.profile_provider is None:
            h_deep = self.default_h_deep_m
            h_point = self.default_h_point_m

            hs_shoaled, ks = self.shoaling_model.transform(hs_offshore, h_deep, h_point)

            depths_default = np.linspace(h_point, h_deep, 50)
            hs_break, h_break = self.breaking_model.apply(hs_offshore, h_deep, depths_default)

            hs_near = min(float(hs_shoaled), float(hs_break))

            # tp_s передаётся в рефракцию — используется в дисперсионном уравнении
            cos_shore, theta_out_deg = self.refraction_model.transform(
                direction_deg=float(direction_deg),
                shore_normal_deg=self.shore_normal_deg,
                h_deep=h_deep,
                h_point=h_point,
                tp_s=tp_s,
            )

            return NearshoreWaveRecord(
                hs_nearshore_m=float(hs_near),
                ks=float(ks),
                h_breaking_m=float(h_break),
                cos_shore=float(cos_shore),
                refracted_angle_deg=float(theta_out_deg),
            )

        # ── Режим с батиметрией ──────────────────────────────────────────────
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

        # tp_s передаётся в рефракцию — используется в дисперсионном уравнении
        cos_shore, theta_out_deg = self.refraction_model.transform(
            direction_deg=float(direction_deg),
            shore_normal_deg=self.shore_normal_deg,
            h_deep=h_deep,
            h_point=h_point,
            tp_s=tp_s,
        )

        return NearshoreWaveRecord(
            hs_nearshore_m=float(hs_near),
            ks=float(ks),
            h_breaking_m=float(h_break),
            cos_shore=float(cos_shore),
            refracted_angle_deg=float(theta_out_deg),
        )
