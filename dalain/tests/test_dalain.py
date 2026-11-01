from dataclasses import replace
import random

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from dalain.geometry import Geometry, InvalidScenarioError, is_connected


def test_original_water_side(anchored_builder):
    g = anchored_builder.geometry
    assert not g.original_water(Point(-5, 0))
    assert g.original_water(Point(5, 0))


def test_internal_hole_rejected(anchored_builder):
    g = anchored_builder.geometry
    ring = frozenset((i, j) for i in range(2, 5) for j in range(3) if (i, j) != (3, 1))
    with pytest.raises(InvalidScenarioError):
        g._topology(ring)


def test_coastal_water_pocket_rejected(anchored_builder):
    g = anchored_builder.geometry
    pocket = frozenset({(i, j) for i in range(5) for j in (-2, 2)}
                       | {(4, j) for j in range(-2, 3)})
    with pytest.raises(InvalidScenarioError):
        g._topology(pocket)


def test_points_do_not_accumulate(anchored_builder):
    b = anchored_builder
    cells = b.initial()
    first, second = b.geometry.scenario(cells), b.geometry.scenario(cells)
    baseline = b.geometry.scenario(frozenset())
    assert first.sites is not second.sites
    assert first.sites == second.sites
    assert len(baseline.sites) == len(b.geometry.base_sites)
    assert first.original_point_count + first.reclamation_point_count == len(first.sites)


@pytest.mark.parametrize("seed", range(30))
def test_normals_and_cell_count(anchored_builder, seed):
    b = anchored_builder
    b.rng = random.Random(seed)
    first = b.initial()
    assert sum(b.geometry.units[c] for c in first) == b.cfg.target_units
    parent = first
    for _ in range(5):
        cells = b.mutate(parent)
        assert is_connected(cells)
        assert b.geometry.valid(cells)
        assert len(cells) == len(first)
        scenario = b.geometry.scenario(cells)
        assert scenario.credited_area_m2 == sum(b.geometry.units[c] for c in cells) * b.cfg.cell_side_m**2 / 2
        assert scenario.original_point_count + scenario.reclamation_point_count == len(scenario.sites)
        land = b.geometry._q(b.geometry.local_land.union(scenario.fill))
        for site in scenario.sites:
            for epsilon in (b.cfg.probe_m, b.cfg.probe_m * 0.1, b.cfg.probe_m * 0.01):
                assert not land.covers(Point(site.x + site.nx * epsilon, site.y + site.ny * epsilon))
                assert land.covers(Point(site.x - site.nx * epsilon, site.y - site.ny * epsilon))
        parent = cells


def test_children_have_no_lakes(anchored_builder):
    b = anchored_builder
    first = b.initial()
    second = b.mutate(first)
    for _ in range(10):
        child = b.child(first, second)
        assert b.geometry.valid(child)
        fill = b.geometry.scenario(child).fill
        polygons = [fill] if fill.geom_type == "Polygon" else fill.geoms
        assert all(not p.interiors for p in polygons)


def test_fragmentation_does_not_multiply_points(anchored_builder, tmp_path):
    original = anchored_builder.geometry
    path = tmp_path / "fragmented.geojson"
    fragments = [LineString([(0, y), (0, y + 20)]) for y in range(-10000, 10000, 20)]
    gpd.GeoDataFrame(geometry=fragments, crs=original.crs).to_crs(4326).to_file(path, driver="GeoJSON")
    fragmented = Geometry(replace(anchored_builder.cfg, coastline_path=path))
    assert len(fragmented.base_sites) == len(original.base_sites)
