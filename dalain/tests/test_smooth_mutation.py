import random
import pytest
from dalain.anchored import perimeter


def test_fraction_rounds_to_even(anchored_builder):
    b = anchored_builder
    for n in (2, 9, 19, 20, 55, 100, 200, 1000):
        cells = frozenset(range(n))
        limit = n // 4
        assert b.change_limit(cells) == limit - limit % 2
        assert b.minimum_change(cells) % 2 == 0


def test_direction_strip_and_perimeter(anchored_builder):
    b = anchored_builder
    parent = b.initial()
    successes = 0
    for seed in range(6):
        b.rng = random.Random(seed + 20)
        child = b.mutate(parent)
        if child == parent:
            continue
        successes += 1
        stats = b.last_stats
        dx, dy = stats["direction_vector"]
        center, width = stats["strip_center_cells"], stats["strip_half_width_cells"]
        for i in range(0, len(b.last_trace), 2):
            removed, added = b.last_trace[i][1], b.last_trace[i + 1][1]
            assert (added[0] - removed[0]) * dx + (added[1] - removed[1]) * dy > 0
            for cell in (removed, added):
                assert abs(-cell[0] * dy + cell[1] * dx - center) <= width + 1e-12
        assert perimeter(child) == stats["perimeter_cells"] <= stats["perimeter_limit_cells"]
        a, c = b.geometry._fill(parent), b.geometry._fill(child)
        assert a.centroid.distance(c.centroid) <= b.centroid_shift(parent) * b.cfg.cell_side_m
        distance = b.boundary_shift(parent) * b.cfg.cell_side_m
        assert c.boundary.difference(a.boundary.buffer(distance)).length <= b.geometry.precision
        assert a.boundary.difference(c.boundary.buffer(distance)).length <= b.geometry.precision
        parent = child
    assert successes > 0


@pytest.mark.parametrize("anchored_builder", [{
    "target_area_m2": 100000.0, "search_radius_m": 800.0, "evaluation_radius_m": 850.0, "max_grid_cells": 30000,
}], indirect=True)
def test_large_shape_reaches_minimum(anchored_builder):
    b = anchored_builder
    parent = b.initial()
    b.rng = random.Random(2)
    child = b.mutate(parent)
    assert len(parent) > 1000
    assert len(child) == len(parent)
    assert b.minimum_change(parent) <= len(parent ^ child) <= b.change_limit(parent)
    assert b.geometry.valid(child)
