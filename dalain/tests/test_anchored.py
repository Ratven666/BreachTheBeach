from dataclasses import replace
import random

import geopandas as gpd
import pytest

from pyproj import CRS
from shapely.geometry import LineString

from dalain.config import Config

from dalain.anchored import (
    AnchoredGeometry,
    GeneticBuilder,
    ANCHOR_CELL,
)


@pytest.fixture
def config(tmp_path):
    crs = CRS.from_proj4(
        "+proj=aeqd "
        "+lat_0=55 "
        "+lon_0=20 "
        "+datum=WGS84 "
        "+units=m"
    )

    path = tmp_path / "coast.geojson"

    gpd.GeoDataFrame(
        geometry=[
            LineString([
                (0, -10000),
                (0, 10000),
            ])
        ],
        crs=crs,
    ).to_crs(4326).to_file(
        path,
        driver="GeoJSON",
    )

    return replace(
        Config(),
        coastline_path=path,
        land_path=None,
        center_lon=20.0,
        center_lat=55.0,
        target_area_m2=1000.0,
        cell_side_m=10.0,
        search_radius_m=150.0,
        evaluation_radius_m=200.0,
        point_spacing_m=20.0,
        tangent_delta_m=30.0,
        probe_m=0.1,
        tolerance_m=0.001,
        max_grid_cells=10000,
        connection="attached",
        sea_side="right",
        construction_attempts=100,
    )


def test_initial_solution_is_anchored(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(42),
    )

    cells = builder.initial()
    scenario = geometry.scenario(cells)

    assert ANCHOR_CELL in cells

    assert scenario.fill.contains(
        geometry.anchor_point
    )

    assert (
        scenario.credited_area_m2
        == config.target_area_m2
    )


def test_initial_is_not_rebuilt(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(42),
    )

    first = builder.initial()
    second = builder.initial()

    assert first == second
    assert first is second


def test_anchor_cannot_be_removed(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(13),
    )

    parent = builder.initial()

    reduced = builder._peel(
        parent,
        len(parent),
    )

    assert ANCHOR_CELL in reduced


def test_mutations_are_local_and_exact(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(42),
    )

    parent = builder.initial()

    for _ in range(30):
        limit = builder.change_limit(
            parent
        )

        child = builder.mutate(parent)

        assert ANCHOR_CELL in child
        assert geometry.valid(child)

        assert len(parent - child) <= limit
        assert len(child - parent) <= 2 * limit

        assert (
            sum(
                geometry.units[cell]
                for cell in child
            )
            == config.target_units
        )

        scenario = geometry.scenario(child)

        assert scenario.fill.contains(
            geometry.anchor_point
        )

        parent = child


def test_donor_does_not_replace_parent(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(91),
    )

    parent = builder.initial()
    donor = builder.mutate(parent)

    limit = builder.change_limit(parent)
    child = builder.mutate(parent, donor)

    assert ANCHOR_CELL in child
    assert geometry.valid(child)

    assert len(parent - child) <= limit
    assert len(child - parent) <= 2 * limit