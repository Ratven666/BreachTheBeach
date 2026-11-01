from dataclasses import replace
import random

import geopandas as gpd
import pytest

from pyproj import CRS

from shapely.geometry import (
    LineString,
    Point,
)

from dalain.config import Config

from dalain.geometry import (
    Geometry,
    InvalidScenarioError,
    is_connected,
)

from dalain.optimize import GeneticBuilder


@pytest.fixture
def config(tmp_path):
    crs = CRS.from_proj4(
        "+proj=aeqd "
        "+lat_0=55 "
        "+lon_0=20 "
        "+datum=WGS84 "
        "+units=m"
    )

    coastline_path = (
        tmp_path / "coast.geojson"
    )

    gpd.GeoDataFrame(
        geometry=[
            LineString([
                (0, -10000),
                (0, 10000),
            ])
        ],
        crs=crs,
    ).to_crs(4326).to_file(
        coastline_path,
        driver="GeoJSON",
    )

    return replace(
        Config(),
        coastline_path=coastline_path,
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


def test_original_water_side(config):
    geometry = Geometry(config)

    assert not geometry.original_water(
        Point(-5, 0)
    )

    assert geometry.original_water(
        Point(5, 0)
    )


def test_internal_hole_rejected(config):
    geometry = Geometry(config)

    ring = frozenset(
        (i, j)
        for i in range(2, 5)
        for j in range(3)
        if (i, j) != (3, 1)
    )

    with pytest.raises(
        InvalidScenarioError
    ):
        geometry._topology(ring)


def test_coastal_water_pocket_rejected(config):
    geometry = Geometry(config)

    pocket = frozenset(
        {
            (i, j)
            for i in range(5)
            for j in (-2, 2)
        }
        | {
            (4, j)
            for j in range(-2, 3)
        }
    )

    with pytest.raises(
        InvalidScenarioError
    ):
        geometry._topology(pocket)


def test_points_do_not_accumulate(config):
    geometry = Geometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(42),
    )

    cells = builder.build()

    first = geometry.scenario(cells)
    second = geometry.scenario(cells)
    baseline = geometry.scenario(frozenset())

    assert first.sites is not second.sites
    assert first.sites == second.sites

    assert len(baseline.sites) == len(
        geometry.base_sites
    )

    assert (
        first.original_point_count
        + first.reclamation_point_count
        == len(first.sites)
    )


@pytest.mark.parametrize(
    "seed",
    range(30),
)
def test_normals_and_exact_area(config, seed):
    geometry = Geometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(seed),
    )

    for _ in range(10):
        cells = builder.build()

        assert is_connected(cells)
        assert geometry.valid(cells)

        scenario = geometry.scenario(cells)

        land = geometry._q(
            geometry.local_land.union(
                scenario.fill
            )
        )

        assert (
            scenario.credited_area_m2
            == config.target_area_m2
        )

        assert (
            scenario.original_point_count
            + scenario.reclamation_point_count
            == len(scenario.sites)
        )

        for site in scenario.sites:
            for epsilon in (
                config.probe_m,
                config.probe_m * 0.1,
                config.probe_m * 0.01,
            ):
                water_probe = Point(
                    site.x + site.nx * epsilon,
                    site.y + site.ny * epsilon,
                )

                land_probe = Point(
                    site.x - site.nx * epsilon,
                    site.y - site.ny * epsilon,
                )

                assert not land.covers(
                    water_probe
                ), (
                    seed,
                    site,
                    epsilon,
                )

                assert land.covers(
                    land_probe
                ), (
                    seed,
                    site,
                    epsilon,
                )


def test_children_have_no_lakes(config):
    geometry = Geometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(51),
    )

    first = builder.build()
    second = builder.build()

    for _ in range(20):
        child = builder.child(
            first,
            second,
        )

        assert geometry.valid(child)

        scenario = geometry.scenario(child)

        polygons = (
            [scenario.fill]
            if scenario.fill.geom_type == "Polygon"
            else scenario.fill.geoms
        )

        assert all(
            not polygon.interiors
            for polygon in polygons
        )


def test_fragmentation_does_not_multiply_points(
    config,
    tmp_path,
):
    original = Geometry(config)

    crs = CRS.from_proj4(
        "+proj=aeqd "
        "+lat_0=55 "
        "+lon_0=20 "
        "+datum=WGS84 "
        "+units=m"
    )

    path = (
        tmp_path / "fragmented.geojson"
    )

    fragments = [
        LineString([
            (0, y),
            (0, y + 20),
        ])
        for y in range(
            -10000,
            10000,
            20,
        )
    ]

    gpd.GeoDataFrame(
        geometry=fragments,
        crs=crs,
    ).to_crs(4326).to_file(
        path,
        driver="GeoJSON",
    )

    fragmented = Geometry(
        replace(
            config,
            coastline_path=path,
        )
    )

    assert (
        len(fragmented.base_sites)
        == len(original.base_sites)
    )