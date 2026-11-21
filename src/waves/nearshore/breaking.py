from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.waves.nearshore.shoaling import ShoalingModel


@dataclass(frozen=True, slots=True)
class BreakingModel:
    gamma_b: float = 0.55
    shoaling_model: ShoalingModel = field(default_factory=ShoalingModel)

    def apply(
        self, hs_offshore: float, h_deep: float, depths_m: np.ndarray
    ) -> tuple[float, float]:
        """
        Определяет высоту и глубину обрушения волны.

        Обход глубин от мелкого к глубокому (берег → море).
        Hs на каждой глубине вычисляется через ShoalingModel (единый источник).
        Первое нарушение критерия Hs >= gamma_b * h — глубина обрушения.
        """
        valid = depths_m[np.isfinite(depths_m) & (depths_m > 0)]
        if valid.size == 0:
            h = 10.0
            return float(min(hs_offshore, self.gamma_b * h)), float(h)

        # Для штатного K_s величина Hs*K_s(h) не растёт с h,
        # если Hs >= 0, а gamma_b*h не убывает. Значит, либо первая
        # глубина подходит, либо не подходит ни одна. Вторая ветка
        # штатного алгоритма всё равно возвращает ту же первую глубину.
        h_break = float(valid.min())
        hs_break = float(self.gamma_b * h_break)
        if (
            type(self.shoaling_model) is ShoalingModel
            and np.isfinite(hs_offshore) and hs_offshore >= 0
            and np.isfinite(h_deep / 0.1)
            and np.isfinite(self.gamma_b) and self.gamma_b >= 0
        ):
            return hs_break, h_break

        # Сохраняем исходную логику для сторонних моделей и нетипичных входов.
        for h in np.sort(valid):
            h_val = float(h)
            # Используем ShoalingModel вместо дублирования формулы
            hs_here, _ = self.shoaling_model.transform(hs_offshore, h_deep, h_val)
            if hs_here >= self.gamma_b * h_val:
                hs_break = self.gamma_b * h_val
                h_break = h_val
                break

        return float(hs_break), float(h_break)
