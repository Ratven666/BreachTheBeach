from __future__ import annotations

from dataclasses import dataclass

from pyproj import Geod

_WGS84 = Geod(ellps="WGS84")


@dataclass
class BathymetryProfileProvider:
    bathymetry_service: object
    origin_lon: float
    origin_lat: float
    radius_m: float = 20_000.0
    n_steps: int = 200

    def get_profile(self, direction_deg: int):
        lon2, lat2, _ = _WGS84.fwd(
            self.origin_lon,
            self.origin_lat,
            float(direction_deg),
            self.radius_m,
        )
        line = self.bathymetry_service.make_line(
            start_lon=self.origin_lon,
            start_lat=self.origin_lat,
            end_lon=lon2,
            end_lat=lat2,
        )
        return self.bathymetry_service.build_profile(line, n_points=self.n_steps)
