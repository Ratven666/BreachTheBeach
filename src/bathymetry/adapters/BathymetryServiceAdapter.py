"""Адаптер между BathymetryService и интерфейсом, ожидаемым WaveClimateService.

WaveClimateService вызывает:
    profile = bathymetry_service.get_profile(direction: int)
    depths   = profile.depths_m   # np.ndarray, глубины от берега к морю

BathymetryService предоставляет:
    profile = service.build_profile(line: GeoLine) -> BathymetryProfile
    depths   = profile.depths      # поле называется depths, не depths_m

Адаптер решает оба несоответствия:
    1. get_profile(direction) → строит GeoLine по азимуту из точки → build_profile()
    2. оборачивает BathymetryProfile в _ProfileWrapper с атрибутом depths_m
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from src.bathymetry.domain.models import GeoLine, GeoPoint
from src.bathymetry.services.BathymetryService import BathymetryService


# ── тонкая обёртка над BathymetryProfile ─────────────────────────────────────

@dataclass
class _ProfileWrapper:
    """Минимальный объект, который видит WaveClimateService._bathy_correction()."""
    depths_m: np.ndarray   # глубины от берега к морю, положительные значения


# ── вспомогательная функция ───────────────────────────────────────────────────

def _destination_point(
    lat: float, lon: float, azimuth_deg: float, distance_m: float
) -> tuple[float, float]:
    """Вычисляет конечную точку по сферической формуле (Vincenty упрощённый).

    Параметры
    ---------
    lat, lon      : исходная точка, градусы
    azimuth_deg   : азимут движения (0 = север, 90 = восток), градусы
    distance_m    : расстояние, метры

    Возвращает (lat2, lon2) в градусах.
    """
    R = 6_371_000.0  # средний радиус Земли, м
    az = math.radians(azimuth_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d_r = distance_m / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_r)
        + math.cos(lat1) * math.sin(d_r) * math.cos(az)
    )
    lon2 = lon1 + math.atan2(
        math.sin(az) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# ── сам адаптер ───────────────────────────────────────────────────────────────

class BathymetryServiceAdapter:
    """Адаптирует BathymetryService под интерфейс, ожидаемый WaveClimateService.

    Использование
    -------------
    adapter = BathymetryServiceAdapter(
        service=bathy_service,   # уже загруженный (после fetch())
        origin_lat=44.72,
        origin_lon=37.82,
        profile_length_m=20_000,  # длина профиля от точки в море
    )

    # передаётся как bathymetry_service= в WaveClimateBatchProcessor.export()
    profile = adapter.get_profile(270)   # азимут 270° = запад
    print(profile.depths_m)              # np.ndarray
    """

    def __init__(
        self,
        service: BathymetryService,
        origin_lat: float,
        origin_lon: float,
        profile_length_m: float = 20_000.0,
        n_points: Optional[int] = None,
    ) -> None:
        if not service.is_loaded:
            raise ValueError(
                "BathymetryService must have a grid loaded (call service.fetch() first)."
            )
        self._service = service
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._profile_length_m = profile_length_m
        self._n_points = n_points
        self._log = logger.bind(cls=self.__class__.__name__)
        self._cache: dict[int, _ProfileWrapper] = {}

    # ── публичный метод — именно его ищет WaveClimateService ─────────────────

    def get_profile(self, direction: int) -> _ProfileWrapper:
        """Строит батиметрический профиль по азимуту от береговой точки.

        Параметры
        ---------
        direction : int
            Азимут в градусах (0–359). Направление «в море» по нормали.

        Возвращает
        ----------
        _ProfileWrapper с атрибутом depths_m (np.ndarray, значения > 0 = под водой).
        """
        direction = int(direction) % 360

        if direction in self._cache:
            return self._cache[direction]

        # конечная точка профиля — уходит от берега в море по азимуту
        end_lat, end_lon = _destination_point(
            self._origin_lat, self._origin_lon,
            azimuth_deg=direction,
            distance_m=self._profile_length_m,
        )

        line = GeoLine(
            start=GeoPoint(lat=self._origin_lat, lon=self._origin_lon),
            end=GeoPoint(lat=end_lat,            lon=end_lon),
        )

        try:
            profile = self._service.build_profile(line, n_points=self._n_points)
            # BathymetryProfile.depths — значения батиметрии (отрицательные = ниже нуля)
            # WaveClimateService ожидает положительные глубины (h > 0 = под водой)
            raw = np.array(profile.depths, dtype=float)
            depths_m = np.where(np.isfinite(raw), np.abs(raw), np.nan)
        except Exception as exc:
            self._log.warning(
                f"build_profile failed for direction={direction}°: {exc}. "
                "Using NaN profile (WaveClimateService will fall back to defaults)."
            )
            depths_m = np.array([np.nan])

        wrapper = _ProfileWrapper(depths_m=depths_m)
        self._cache[direction] = wrapper
        return wrapper
