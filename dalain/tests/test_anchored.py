import pytest
from dalain.anchored import ANCHOR_CELL


def test_initial_is_fixed_and_exact(anchored_builder):
    b = anchored_builder
    cells = b.initial()
    assert b.initial() is cells
    assert sum(b.geometry.units[c] for c in cells) == b.cfg.target_units
    assert b.geometry.fixed_cell_count == len(cells)
    assert ANCHOR_CELL in cells
    assert b.geometry.valid(cells)
    assert b.geometry.scenario(cells).fill.contains(b.geometry.anchor_point)


def test_anchor_and_second_lock_are_rejected(anchored_builder):
    b = anchored_builder
    cells = b.initial()
    with pytest.raises(ValueError):
        b.geometry.lock_cell_count(len(cells) + 1)
    assert not b.geometry.valid(cells - {ANCHOR_CELL})


def test_small_budget_is_explicit(anchored_builder):
    b = anchored_builder
    for n in range(2, 8):
        cells = frozenset(range(n))
        assert b.change_limit(cells) == 0
        assert b.minimum_change(cells) >= 2


@pytest.mark.parametrize("anchored_builder", [{"connection": "island", "coast_x": -150}], indirect=True)
def test_island_invariants(anchored_builder):
    b = anchored_builder
    parent = b.initial()
    child = b.mutate(parent)
    assert len(child) == len(parent)
    assert all(b.geometry.units[c] == 2 for c in child)
    assert b.geometry.valid(child)
    assert not b.geometry._fill(child).intersects(b.geometry.local_land)
