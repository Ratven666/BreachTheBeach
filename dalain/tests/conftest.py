from dataclasses import replace
import random

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString

from dalain.config import Config
from dalain.anchored import AnchoredGeometry, GeneticBuilder


@pytest.fixture
def anchored_builder(tmp_path, request):
    crs = CRS.from_proj4("+proj=aeqd +lat_0=55 +lon_0=20 +datum=WGS84 +units=m")
    options = dict(getattr(request, "param", {}))
    coast_x = options.pop("coast_x", 0)
    path = tmp_path / "coast.geojson"
    gpd.GeoDataFrame(geometry=[LineString([(coast_x, -10000), (coast_x, 10000)])], crs=crs).to_crs(4326).to_file(
        path, driver="GeoJSON")
    cfg = replace(Config(), coastline_path=path, land_path=None,
                  center_lon=20.0, center_lat=55.0, target_area_m2=5000.0,
                  cell_side_m=10.0, search_radius_m=250.0, evaluation_radius_m=300.0,
                  point_spacing_m=20.0, construction_attempts=100, max_grid_cells=10000)
    cfg = replace(cfg, **options)
    geometry = AnchoredGeometry(cfg)
    return GeneticBuilder(cfg, geometry, random.Random(42))
