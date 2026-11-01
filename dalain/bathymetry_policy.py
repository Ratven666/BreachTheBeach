from dataclasses import dataclass
import math


# ============================================================
# ОБРАБОТКА ОТСУТСТВУЮЩЕЙ БАТИМЕТРИИ
# ============================================================

# "default_depths":
#   использовать явно заданную модель постоянных глубин;
#   не выдавать её за измеренную батиметрию.
#
# "repository_zero":
#   поведение исходного _process_point():
#   при отсутствии глубин CWEF затронутых дней равен нулю.
#
# "strict":
#   остановить расчёт.
MISSING_BATHYMETRY_POLICY = "default_depths"


# Глубины модели без измеренного профиля.
# Аналог параметров default_h_deep_m / default_h_point_m
# в NearshoreWaveTransformer исходного репозитория.
FALLBACK_REFERENCE_DEPTH_M = 20.0
FALLBACK_COASTAL_DEPTH_M = 3.0


@dataclass(frozen=True)
class BathymetryPolicy:
    mode: str = MISSING_BATHYMETRY_POLICY
    reference_depth_m: float = FALLBACK_REFERENCE_DEPTH_M
    coastal_depth_m: float = FALLBACK_COASTAL_DEPTH_M

    def validate(self):
        if self.mode not in {
            "default_depths",
            "repository_zero",
            "strict",
        }:
            raise ValueError(
                "Политика батиметрии должна быть одной из: "
                "default_depths, repository_zero, strict"
            )

        for name, value in {
            "reference_depth_m": self.reference_depth_m,
            "coastal_depth_m": self.coastal_depth_m,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} должно быть конечным числом > 0"
                )


POLICY = BathymetryPolicy()