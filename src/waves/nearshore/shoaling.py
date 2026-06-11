from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShoalingModel:
    def coefficient(self, h_deep: float, h_point: float) -> float:
        h_ref = max(float(h_point), 0.1)
        return float((float(h_deep) / h_ref) ** 0.25)

    def transform(self, hs_offshore: float, h_deep: float, h_point: float) -> tuple[float, float]:
        ks = self.coefficient(h_deep, h_point)
        return float(hs_offshore * ks), float(ks)
