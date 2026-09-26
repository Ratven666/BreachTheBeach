from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.coastline.storage.WindFetchRepository import (
    WindFetchRepository,
)
from src.coastline.storage.models import (
    Base,
    CoastlinePointModel,
    CoastlineSourceModel,
    WindFetchModel,
)
from src.wind_fetch.models import (
    MultiDirectionFetchResult,
)


def make_result(
    point_id: int,
    azimuth_deg: float,
    fetch_length_m: float,
) -> MultiDirectionFetchResult:
    return MultiDirectionFetchResult(
        point_id=point_id,
        direction_id=int(azimuth_deg) + 1,
        normal_azimuth_deg=0.0,
        azimuth_deg=azimuth_deg,
        source_point_lon=30.0,
        source_point_lat=60.0,
        start_point_lon=30.0,
        start_point_lat=60.0,
        fetch_length_m=fetch_length_m,
        hit_found=True,
        hit_lon=30.1,
        hit_lat=60.1,
        used_default_value=False,
        skipped_by_land_sector=False,
    )


def create_test_session() -> tuple[Session, int]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:"
    )

    Base.metadata.create_all(engine)

    session = Session(engine)

    source = CoastlineSourceModel(
        name="test",
        points_count=1,
    )
    session.add(source)
    session.flush()

    point = CoastlinePointModel(
        source_id=source.id,
        seq=0,
        lon=30.0,
        lat=60.0,
    )
    session.add(point)
    session.commit()

    return session, point.id


def test_schema_is_created_with_main_metadata() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:"
    )

    Base.metadata.create_all(engine)

    assert "wind_fetches" in Base.metadata.tables


def test_replace_for_points_writes_rows() -> None:
    session, point_id = create_test_session()
    repository = WindFetchRepository(session)

    saved_count = repository.replace_for_points(
        [
            make_result(
                point_id=point_id,
                azimuth_deg=0.0,
                fetch_length_m=1_000.0,
            ),
            make_result(
                point_id=point_id,
                azimuth_deg=90.0,
                fetch_length_m=2_000.0,
            ),
        ]
    )
    session.commit()

    rows = (
        session.execute(
            select(WindFetchModel).order_by(
                WindFetchModel.azimuth_deg
            )
        )
        .scalars()
        .all()
    )

    assert saved_count == 2
    assert len(rows) == 2

    assert rows[0].point_id == point_id
    assert rows[0].azimuth_deg == 0.0
    assert rows[0].fetch_length_m == 1_000.0

    assert rows[1].point_id == point_id
    assert rows[1].azimuth_deg == 90.0
    assert rows[1].fetch_length_m == 2_000.0

    session.close()


def test_replace_for_points_replaces_existing_rows() -> None:
    session, point_id = create_test_session()
    repository = WindFetchRepository(session)

    repository.replace_for_points(
        [
            make_result(
                point_id=point_id,
                azimuth_deg=0.0,
                fetch_length_m=1_000.0,
            ),
            make_result(
                point_id=point_id,
                azimuth_deg=90.0,
                fetch_length_m=2_000.0,
            ),
        ]
    )
    session.commit()

    saved_count = repository.replace_for_points(
        [
            make_result(
                point_id=point_id,
                azimuth_deg=180.0,
                fetch_length_m=3_000.0,
            ),
        ]
    )
    session.commit()

    rows = (
        session.execute(
            select(WindFetchModel)
        )
        .scalars()
        .all()
    )

    assert saved_count == 1
    assert len(rows) == 1

    assert rows[0].point_id == point_id
    assert rows[0].azimuth_deg == 180.0
    assert rows[0].fetch_length_m == 3_000.0

    session.close()


def test_replace_for_points_rejects_unknown_point() -> None:
    session, _ = create_test_session()
    repository = WindFetchRepository(session)

    with pytest.raises(
        ValueError,
        match="999",
    ):
        repository.replace_for_points(
            [
                make_result(
                    point_id=999,
                    azimuth_deg=0.0,
                    fetch_length_m=1_000.0,
                )
            ]
        )

    session.close()


def test_replace_for_points_rejects_duplicate_direction() -> None:
    session, point_id = create_test_session()
    repository = WindFetchRepository(session)

    with pytest.raises(
        ValueError,
        match="Duplicate wind-fetch result",
    ):
        repository.replace_for_points(
            [
                make_result(
                    point_id=point_id,
                    azimuth_deg=90.0,
                    fetch_length_m=1_000.0,
                ),
                make_result(
                    point_id=point_id,
                    azimuth_deg=90.0,
                    fetch_length_m=2_000.0,
                ),
            ]
        )

    session.close()


def test_load_for_point_orders_rows_by_azimuth() -> None:
    session, point_id = create_test_session()
    repository = WindFetchRepository(session)

    repository.replace_for_points(
        [
            make_result(
                point_id=point_id,
                azimuth_deg=270.0,
                fetch_length_m=3_000.0,
            ),
            make_result(
                point_id=point_id,
                azimuth_deg=0.0,
                fetch_length_m=1_000.0,
            ),
            make_result(
                point_id=point_id,
                azimuth_deg=90.0,
                fetch_length_m=2_000.0,
            ),
        ]
    )
    session.commit()

    rows = repository.load_for_point(point_id)

    assert [
        row.azimuth_deg
        for row in rows
    ] == [
        0.0,
        90.0,
        270.0,
    ]

    session.close()
