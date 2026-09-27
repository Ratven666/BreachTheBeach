from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.base.BBox import BBox
from src.bathymetry.archive import (
    BathymetryArchive,
    BathymetryArchiveBuilder,
    BathymetryArchiveConfig,
    BathymetryCoveragePlanner,
)
from src.bathymetry.domain.models import (
    BathymetryGrid,
)
from src.bathymetry.loaders.BathymetryLoader import (
    BathymetryLoader,
)
from src.coastline.storage.models import (
    Base,
    CoastlinePointModel,
    CoastlineSourceModel,
    WindFetchModel,
)


class FakeLoader(BathymetryLoader):
    """
    Детерминированный загрузчик для тестов.

    Он не обращается к внешней сети и создаёт постоянную
    глубину -12.5 м во всём переданном bbox.
    """

    @property
    def source_name(self) -> str:
        return "fake"

    def load(
        self,
        bbox: BBox,
    ) -> BathymetryGrid:
        lats = np.linspace(
            bbox.south,
            bbox.north,
            41,
        )

        lons = np.linspace(
            bbox.west,
            bbox.east,
            41,
        )

        return BathymetryGrid(
            lats=lats,
            lons=lons,
            z=np.full(
                (41, 41),
                -12.5,
            ),
            source=self.source_name,
            resolution_arcsec=3.75,
        )


def make_session(
    *,
    with_missing_fetch: bool = False,
) -> tuple[
    Session,
    list[int],
]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:"
    )

    Base.metadata.create_all(
        engine
    )

    session = Session(
        engine
    )

    source = CoastlineSourceModel(
        name="test",
        crs="EPSG:4326",
        points_count=6,
    )

    session.add(
        source
    )

    session.flush()

    point_ids: list[int] = []

    for seq in range(6):
        point = CoastlinePointModel(
            source_id=source.id,
            seq=seq,
            lon=37.70 + seq * 0.08,
            lat=44.70 + seq * 0.01,
        )

        session.add(
            point
        )

        session.flush()

        point_ids.append(
            point.id
        )

        if (
            with_missing_fetch
            and seq == 5
        ):
            continue

        for (
            azimuth,
            length,
        ) in (
            (0.0, 12_000),
            (90.0, 18_000),
            (180.0, 15_000),
        ):
            session.add(
                WindFetchModel(
                    point_id=point.id,
                    azimuth_deg=azimuth,
                    fetch_length_m=length,
                )
            )

    session.commit()

    return session, point_ids


def test_planner_covers_each_fetch_radius_and_uses_overlap(
    tmp_path: Path,
) -> None:
    session, point_ids = (
        make_session()
    )

    config = BathymetryArchiveConfig(
        output_dir=tmp_path,
        max_cells_per_tile=15_000,
        planning_resolution_arcsec=15.0,
        overlap_points=1,
        padding_m=500.0,
    )

    (
        source_id,
        points,
        tiles,
    ) = BathymetryCoveragePlanner(
        session,
        config,
    ).plan()

    assert source_id == 1

    assert [
        point.point_id
        for point in points
    ] == point_ids

    assert len(tiles) > 1

    # Соседние тайлы должны содержать хотя бы одну общую
    # береговую точку.
    assert (
        set(tiles[0].point_ids)
        & set(tiles[1].point_ids)
    )

    # Для каждой точки должен существовать тайл, полностью
    # покрывающий рассчитанный required_bbox.
    for point in points:
        containing_tiles = [
            tile
            for tile in tiles
            if point.point_id
            in tile.point_ids
        ]

        assert containing_tiles

        assert any(
            tile.required_bbox.south
            <= point.required_bbox.south
            and tile.required_bbox.west
            <= point.required_bbox.west
            and tile.required_bbox.north
            >= point.required_bbox.north
            and tile.required_bbox.east
            >= point.required_bbox.east
            for tile in containing_tiles
        )

    session.close()


def test_planner_rejects_points_without_fetches(
    tmp_path: Path,
) -> None:
    session, _ = make_session(
        with_missing_fetch=True
    )

    config = BathymetryArchiveConfig(
        output_dir=tmp_path
    )

    with pytest.raises(
        ValueError,
        match="Wind fetches are missing",
    ):
        BathymetryCoveragePlanner(
            session,
            config,
        ).plan(
            1
        )

    session.close()


def test_builder_creates_manifest_and_tiles(
    tmp_path: Path,
) -> None:
    session, point_ids = (
        make_session()
    )

    config = BathymetryArchiveConfig(
        output_dir=tmp_path,
        max_cells_per_tile=10_000_000,
        planning_resolution_arcsec=15.0,
        padding_m=1_000.0,
        retries=1,
    )

    builder = BathymetryArchiveBuilder(
        session=session,
        config=config,
        loader_factory=(
            lambda bbox: FakeLoader()
        ),
    )

    manifest_path = builder.build(
        1
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["point_source_id"]
        == 1
    )

    assert set(
        manifest["points"]
    ) == {
        str(point_id)
        for point_id in point_ids
    }

    assert manifest["tiles"]

    assert all(
        (
            tmp_path
            / tile["file"]
        ).exists()
        for tile in manifest["tiles"]
    )

    session.close()


def test_archive_profile_is_local_and_has_positive_depths(
    tmp_path: Path,
) -> None:
    session, point_ids = (
        make_session()
    )

    config = BathymetryArchiveConfig(
        output_dir=tmp_path,
        max_cells_per_tile=10_000_000,
        planning_resolution_arcsec=15.0,
        padding_m=1_000.0,
        retries=1,
    )

    manifest_path = (
        BathymetryArchiveBuilder(
            session=session,
            config=config,
            loader_factory=(
                lambda bbox: FakeLoader()
            ),
        ).build(
            1
        )
    )

    archive = BathymetryArchive(
        manifest_path
    )

    point_service = (
        archive.for_point_id(
            point_ids[0],
            radius_m=10_000,
            n_steps=25,
        )
    )

    profile = (
        point_service.get_profile(
            90
        )
    )

    assert profile.direction == 90

    assert len(
        profile.distances_m
    ) == 25

    assert (
        profile.distances_m[-1]
        == pytest.approx(
            10_000,
            rel=1e-5,
        )
    )

    assert np.allclose(
        profile.depths_m,
        12.5,
    )

    # Повторный запрос использует кэш профиля.
    assert (
        point_service.get_profile(90)
        is profile
    )

    session.close()


def test_archive_resolves_point_by_coordinates(
    tmp_path: Path,
) -> None:
    session, point_ids = (
        make_session()
    )

    config = BathymetryArchiveConfig(
        output_dir=tmp_path,
        max_cells_per_tile=10_000_000,
        planning_resolution_arcsec=15.0,
        retries=1,
    )

    manifest_path = (
        BathymetryArchiveBuilder(
            session=session,
            config=config,
            loader_factory=(
                lambda bbox: FakeLoader()
            ),
        ).build(
            1
        )
    )

    archive = BathymetryArchive(
        manifest_path
    )

    metadata = archive.point_metadata(
        point_ids[0]
    )

    point_service = archive.for_point(
        lon=metadata["lon"],
        lat=metadata["lat"],
        n_steps=20,
    )

    assert (
        point_service.point_id
        == point_ids[0]
    )

    session.close()


def test_archive_rejects_radius_above_database_maximum(
    tmp_path: Path,
) -> None:
    session, point_ids = (
        make_session()
    )

    config = BathymetryArchiveConfig(
        output_dir=tmp_path,
        max_cells_per_tile=10_000_000,
        planning_resolution_arcsec=15.0,
        retries=1,
    )

    manifest_path = (
        BathymetryArchiveBuilder(
            session=session,
            config=config,
            loader_factory=(
                lambda bbox: FakeLoader()
            ),
        ).build(
            1
        )
    )

    archive = BathymetryArchive(
        manifest_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "exceeds archived "
            "max_fetch_m"
        ),
    ):
        archive.for_point_id(
            point_ids[0],
            radius_m=18_001,
        )

    session.close()
