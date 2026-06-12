from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShoalingModel:
    def coefficient(self, h_deep: float, h_point: float) -> float:
        """Коэффициент осушения Ks = (h_deep / h_point)^0.25.

        Защита: h_deep не может быть меньше h_point физически.
        Если батиметрия содержит инвертированный профиль (точка наблюдения
        глубже опорной), зажимаем h_deep снизу до h_point, чтобы Ks ≤ 1
        и волна не «усиливалась» без физического смысла.
        """
        h_ref = max(float(h_point), 0.1)
        h_deep_safe = max(float(h_deep), h_ref)   # ← исправлено
        return float((h_deep_safe / h_ref) ** 0.25)

    def transform(
        self, hs_offshore: float, h_deep: float, h_point: float
    ) -> tuple[float, float]:
        ks = self.coefficient(h_deep, h_point)
        return float(hs_offshore * ks), float(ks)
