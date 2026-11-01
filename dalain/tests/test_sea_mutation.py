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
        target_area_m2=5000.0,
        cell_side_m=10.0,
        search_radius_m=250.0,
        evaluation_radius_m=300.0,
        point_spacing_m=20.0,
        tangent_delta_m=30.0,
        probe_m=0.1,
        tolerance_m=0.001,
        max_grid_cells=10000,
        connection="attached",
        sea_side="right",
        construction_attempts=100,
    )


def test_every_change_uses_current_sea_front(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(42),
    )

    parent = builder.initial()
    changed_count = 0

    for _ in range(10):
        child = builder.mutate(parent)

        current = set(parent)
        removed = set()
        added = set()

        for action, cell in builder.last_trace:
            removable, addable = (
                builder._sea_frontiers(
                    current
                )
            )

            if action == "remove":
                assert cell in removable
                assert cell != ANCHOR_CELL

                current.remove(cell)
                removed.add(cell)

            elif action == "add":
                assert cell in addable

                # Удалённые клетки не возвращаются
                # в рамках той же мутации.
                assert cell not in removed

                current.add(cell)
                added.add(cell)

            else:
                raise AssertionError(
                    f"Неизвестное действие: {action}"
                )

        assert frozenset(current) == child
        assert not (removed & added)

        assert ANCHOR_CELL in child
        assert geometry.valid(child)

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

        limit = builder.change_limit(parent)

        assert len(parent - child) <= limit
        assert len(child - parent) <= 2 * limit

        if child != parent:
            changed_count += 1

            assert builder.last_trace
            assert removed
            assert added

        parent = child

    assert changed_count > 0


def test_sea_front_does_not_include_anchor_removal(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(7),
    )

    cells = builder.initial()

    removable, addable = (
        builder._sea_frontiers(cells)
    )

    assert ANCHOR_CELL not in removable

    assert all(
        cell in cells
        for cell in removable
    )

    assert all(
        cell not in cells
        for cell in addable
    )


def test_initial_solution_is_not_rebuilt(config):
    geometry = AnchoredGeometry(config)

    builder = GeneticBuilder(
        config,
        geometry,
        random.Random(19),
    )

    first = builder.initial()
    second = builder.initial()

    assert first is second