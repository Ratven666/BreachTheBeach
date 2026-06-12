from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BreakingModel:
    gamma_b: float = 0.55

    def apply(self, hs_offshore: float, h_deep: float, depths_m: np.ndarray) -> tuple[float, float]:
        valid = depths_m[np.isfinite(depths_m) & (depths_m > 0)]
        if valid.size == 0:
            h = 10.0
            return float(min(hs_offshore, self.gamma_b * h)), float(h)

        hs_break = float(hs_offshore)
        h_break = float(valid[0])
        for h in np.sort(valid)[::-1]:
            h_val = float(h)
            h_here = float(hs_offshore) * (float(h_deep) / h_val) ** 0.25
            if h_here >= self.gamma_b * h_val:
                hs_break = self.gamma_b * h_val
                h_break = h_val
                break
        return float(hs_break), float(h_break)
