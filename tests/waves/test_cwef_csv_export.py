from __future__ import annotations

"""
Автономные тесты экспортёра CWEF.

Не используют:
- реальные береговые GeoJSON;
- реальные SQLite-базы;
- файлы с результатами предыдущих запусков;
- сетевые сервисы.

Все входные данные создаются во временных каталогах pytest.

Проверяемый модуль:
    Kaliningrad/5_8_export_cwef_by_coastline.py

Если он находится в db_pypeline/, используется этот путь.

Тесты рассчитаны на версию экспортёра с функциями:
    read_features
    metric_crs
    build_components
    associate
    export

и классами:
    CoastFeature
    WavePoint
"""

import csv
import importlib.util
import json
import sqlite3
import sys

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

from pyproj import Transformer
from shapely.geometry import LineString, Point


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODULE_CANDIDATES = (
    PROJECT_ROOT
    / "Kaliningrad"
    / "5_8_export_cwef_by_coastline.py",

    PROJECT_ROOT
    / "db_pypeline"
    / "5_8_export_cwef_by_coastline.py",
)

TEST_CRS = "EPSG:3857"

TO_WGS84 = Transformer.from_crs(
    TEST_CRS,
    "EPSG:4326",
    always_xy=True,
)

# Контролируемый тестовый период содержит високосный день.
TEST_START = date(2000, 2, 28)
TEST_END = date(2000, 3, 1)
TEST_EPOCH = date(1940, 1, 1)
TEST_FIRST_DAY = (TEST_START - TEST_EPOCH).days

EXPECTED_DATES = [
    "2000-02-28",
    "2000-02-29",
    "2000-03-01",
]


# Самодостаточная тестовая схема.
# Содержит таблицы и поля, используемые экспортёром.
# Реальная база проекта не читается.
TEST_SCHEMA_SQL = """
PRAGMA user_version = 3;
PRAGMA foreign_keys = ON;

CREATE TABLE wave_points (
    id INTEGER PRIMARY KEY,
    lon REAL NOT NULL,
    lat REAL NOT NULL,
    normal_azimuth_deg INTEGER NOT NULL
);

CREATE TABLE wave_activity_summary (
    point_id INTEGER PRIMARY KEY,
    n_days INTEGER NOT NULL CHECK (n_days >= 0),
    FOREIGN KEY (point_id) REFERENCES wave_points(id)
);

CREATE TABLE wave_activity (
    point_id INTEGER NOT NULL,
    day INTEGER NOT NULL,
    wind_azimuth_deg INTEGER NOT NULL,
    cwef_wm INTEGER NOT NULL CHECK (cwef_wm >= 1),
    PRIMARY KEY (point_id, day),
    FOREIGN KEY (point_id) REFERENCES wave_points(id)
);
"""


@dataclass(frozen=True)
class TestPaths:
    geojson: Path
    database: Path
    output: Path


def load_exporter() -> ModuleType:
    exporter_path = next(
        (
            path
            for path in MODULE_CANDIDATES
            if path.is_file()
        ),
        None,
    )

    if exporter_path is None:
        checked = "\n".join(
            str(path)
            for path in MODULE_CANDIDATES
        )

        raise FileNotFoundError(
            "Не найден проверяемый модуль экспортёра.\n"
            f"Проверены пути:\n{checked}"
        )

    module_name = "_cwef_exporter_under_test"

    spec = importlib.util.spec_from_file_location(
        module_name,
        exporter_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Не удалось загрузить экспортёр: {exporter_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Нужно в том числе для корректной работы dataclass
    # при динамическом импорте.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="session")
def exporter() -> ModuleType:
    return load_exporter()


@pytest.fixture
def sandbox(
    exporter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestPaths:
    paths = TestPaths(
        geojson=tmp_path / "coastlines.geojson",
        database=tmp_path / "wave_activity.db",
        output=tmp_path / "results",
    )

    settings = {
        "COASTLINES_GEOJSON": paths.geojson,
        "WAVE_ACTIVITY_DB": paths.database,
        "OUTPUT_DIR": paths.output,
        "START_DATE": TEST_START,
        "END_DATE": TEST_END,
        "INPUT_CRS": TEST_CRS,
        "WORKING_CRS": TEST_CRS,
        "ENDPOINT_JOIN_TOLERANCE_M": 0.01,
        "MAX_DISTANCE_TO_LINE_M": 1.0,
        "DISTANCE_TIE_TOLERANCE_M": 0.000001,
        "AMBIGUOUS_POINT_POLICY": "first",
        "AMBIGUOUS_COMPONENT_POLICY": "first",
        "START_ANCHORS_WGS84": {},
        "REVERSE_COMPONENTS": set(),
        "START_ANCHOR_TOLERANCE_M": 2.0,
        "MISSING_DAY_POLICY": "strict",
        "CSV_DELIMITER": ",",
        "CSV_ENCODING": "utf-8",
        "DAY_CHUNK_SIZE": 2,
        "SQL_POINT_BATCH_SIZE": 2,
        "MAX_MATRIX_MIB": 1,
        "OVERWRITE": False,
    }

    for name, value in settings.items():
        monkeypatch.setattr(
            exporter,
            name,
            value,
        )

    return paths


def wgs84(x: float, y: float) -> tuple[float, float]:
    return TO_WGS84.transform(x, y)


def make_feature(
    parts: list[list[tuple[float, float]]],
    line_id: int = 0,
) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "line_id": line_id,
        },
        "geometry": {
            "type": "MultiLineString",
            "coordinates": parts,
        },
    }


def write_geojson(
    path: Path,
    features: list[dict],
) -> None:
    document = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": TEST_CRS,
            },
        },
        "features": features,
    }

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )


def create_database(
    path: Path,
    points: list[tuple[int, float, float, int | None]],
    activity: tuple[tuple[int, int, int], ...] = (),
) -> None:
    """
    points:
        (point_id, x_метры, y_метры, n_days)

    n_days=None:
        запись summary отсутствует.

    activity:
        (point_id, смещение_от_TEST_START, CWEF)
    """
    with sqlite3.connect(path) as connection:
        connection.executescript(TEST_SCHEMA_SQL)

        for point_id, x, y, n_days in points:
            lon, lat = wgs84(x, y)

            connection.execute(
                """
                INSERT INTO wave_points (
                    id, lon, lat, normal_azimuth_deg
                )
                VALUES (?, ?, ?, ?)
                """,
                (point_id, lon, lat, 270),
            )

            if n_days is not None:
                connection.execute(
                    """
                    INSERT INTO wave_activity_summary (
                        point_id, n_days
                    )
                    VALUES (?, ?)
                    """,
                    (point_id, n_days),
                )

        connection.executemany(
            """
            INSERT INTO wave_activity (
                point_id, day, wind_azimuth_deg, cwef_wm
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    point_id,
                    TEST_FIRST_DAY + offset,
                    270,
                    value,
                )
                for point_id, offset, value in activity
            ],
        )


def read_csv(path: Path) -> list[list[str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.reader(stream))


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def single_line_feature() -> dict:
    return make_feature(
        [
            [
                (0.0, 1000.0),
                (30.0, 1000.0),
            ],
        ]
    )


def prepare_single_line(
    paths: TestPaths,
    *,
    n_days: int | None = 3,
    activity: tuple[tuple[int, int, int], ...] = (),
) -> None:
    write_geojson(
        paths.geojson,
        [single_line_feature()],
    )

    create_database(
        paths.database,
        [(7, 10.0, 1000.0, n_days)],
        activity,
    )


def test_export_keeps_one_set_per_feature(
    exporter,
    sandbox,
):
    """
    Первый Feature:
        три соединённые части и одна отдельная компонента.

    Второй Feature:
        замкнутая линия.

    Ожидается два комплекта, а не пять.
    """
    first_feature = make_feature(
        [
            [(10.0, 1000.0), (20.0, 1000.0)],
            [(30.0, 1000.0), (20.0, 1000.0)],
            [(10.0, 1000.0), (0.0, 1000.0)],
            [(100.0, 1000.0), (120.0, 1000.0)],
        ],
        line_id=100,
    )

    second_feature = make_feature(
        [
            [
                (200.0, 1000.0),
                (220.0, 1000.0),
                (220.0, 1020.0),
                (200.0, 1020.0),
                (200.0, 1000.0),
            ]
        ],
        line_id=200,
    )

    write_geojson(
        sandbox.geojson,
        [first_feature, second_feature],
    )

    create_database(
        sandbox.database,
        [
            (90, 5.0, 1000.0, 3),
            (2, 25.0, 1000.0, 3),
            (70, 110.0, 1000.0, 3),
            (50, 210.0, 1000.0, 3),
        ],
        (
            (90, 0, 12),
            (2, 1, 30),
            (70, 2, 60),
            (50, 1, 9),
        ),
    )

    exporter.export()

    assert len(
        list(sandbox.output.glob("coastline_*_cwef.csv"))
    ) == 2

    assert len(
        list(sandbox.output.glob("coastline_*_points.csv"))
    ) == 2

    assert len(
        list(sandbox.output.glob("coastline_*_points.geojson"))
    ) == 2

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert values == [
        ["date", "90", "2", "70"],
        ["2000-02-28", "12", "0", "0"],
        ["2000-02-29", "0", "30", "0"],
        ["2000-03-01", "0", "0", "60"],
    ]

    points = read_dict_csv(
        sandbox.output / "coastline_0001_points.csv"
    )

    assert [
        int(row["component_id"])
        for row in points
    ] == [1, 1, 2]

    assert [
        float(row["chainage_component_m"])
        for row in points
    ] == pytest.approx([5.0, 25.0, 10.0])

    assert [
        float(row["chainage_feature_excluding_gaps_m"])
        for row in points
    ] == pytest.approx([5.0, 25.0, 40.0])

    report = read_json(
        sandbox.output / "export_geometry_report.json"
    )

    assert report["feature_count"] == 2
    assert report["component_count"] == 3


def test_start_is_true_endpoint_not_start_of_middle_part(
    exporter,
    sandbox,
):
    write_geojson(
        sandbox.geojson,
        [
            make_feature(
                [
                    [(10.0, 1000.0), (20.0, 1000.0)],
                    [(30.0, 1000.0), (20.0, 1000.0)],
                    [(10.0, 1000.0), (0.0, 1000.0)],
                ]
            )
        ],
    )

    features = exporter.read_features(sandbox.geojson)
    crs = exporter.metric_crs(features)
    components = exporter.build_components(features, crs)

    assert len(components) == 1

    geometry = components[0].geometry

    assert Point(geometry.coords[0]).distance(
        Point(0.0, 1000.0)
    ) < 0.000001

    assert Point(geometry.coords[-1]).distance(
        Point(30.0, 1000.0)
    ) < 0.000001

    assert geometry.length == pytest.approx(30.0)


def test_csv_and_geojson_have_identical_ids_and_order(
    exporter,
    sandbox,
):
    write_geojson(
        sandbox.geojson,
        [single_line_feature()],
    )

    create_database(
        sandbox.database,
        [
            (100, 5.0, 1000.0, 3),
            (1, 15.0, 1000.0, 3),
            (50, 25.0, 1000.0, 3),
        ],
    )

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    points = read_dict_csv(
        sandbox.output / "coastline_0001_points.csv"
    )

    geojson = read_json(
        sandbox.output / "coastline_0001_points.geojson"
    )

    value_ids = [
        int(value)
        for value in values[0][1:]
    ]

    point_ids = [
        int(row["point_id"])
        for row in points
    ]

    geojson_ids = [
        feature["properties"]["point_id"]
        for feature in geojson["features"]
    ]

    assert value_ids == point_ids == geojson_ids
    assert value_ids == [100, 1, 50]


def test_mean_includes_zero_days(
    exporter,
    sandbox,
):
    prepare_single_line(
        sandbox,
        activity=((7, 1, 30),),
    )

    exporter.export()

    geojson = read_json(
        sandbox.output / "coastline_0001_points.geojson"
    )

    properties = geojson["features"][0]["properties"]

    assert properties["mean_cwef_wm"] == pytest.approx(10.0)
    assert properties["n_values_for_mean"] == 3
    assert properties["n_missing_days"] == 0


@pytest.mark.parametrize("n_days", [0, None])
def test_failed_or_missing_summary_is_not_forged_as_zero(
    exporter,
    sandbox,
    n_days,
):
    prepare_single_line(
        sandbox,
        n_days=n_days,
    )

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert [
        row[1]
        for row in values[1:]
    ] == ["", "", ""]

    geojson = read_json(
        sandbox.output / "coastline_0001_points.geojson"
    )

    properties = geojson["features"][0]["properties"]

    assert properties["mean_cwef_wm"] is None
    assert properties["n_values_for_mean"] == 0
    assert properties["n_missing_days"] == 3


def test_incomplete_series_strict_policy_raises(
    exporter,
    sandbox,
):
    prepare_single_line(
        sandbox,
        n_days=2,
        activity=((7, 1, 9),),
    )

    with pytest.raises(
        ValueError,
        match="даты пропусков неизвестны",
    ):
        exporter.export()

    assert not sandbox.output.exists()


@pytest.mark.parametrize(
    ("policy", "expected_values", "expected_mean", "expected_count"),
    [
        ("nan", ["", "9", ""], 9.0, 1),
        ("zero", ["0", "9", "0"], 3.0, 3),
    ],
)
def test_incomplete_series_explicit_policy(
    exporter,
    sandbox,
    monkeypatch,
    policy,
    expected_values,
    expected_mean,
    expected_count,
):
    monkeypatch.setattr(
        exporter,
        "MISSING_DAY_POLICY",
        policy,
    )

    prepare_single_line(
        sandbox,
        n_days=2,
        activity=((7, 1, 9),),
    )

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert [
        row[1]
        for row in values[1:]
    ] == expected_values

    geojson = read_json(
        sandbox.output / "coastline_0001_points.geojson"
    )

    properties = geojson["features"][0]["properties"]

    assert properties["mean_cwef_wm"] == pytest.approx(
        expected_mean
    )

    assert properties["n_values_for_mean"] == expected_count


def test_point_assigned_to_only_one_feature_on_tie(
    exporter,
    sandbox,
):
    feature = single_line_feature()

    write_geojson(
        sandbox.geojson,
        [feature, feature],
    )

    create_database(
        sandbox.database,
        [(7, 10.0, 1000.0, 3)],
    )

    exporter.export()

    first = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    second = read_csv(
        sandbox.output / "coastline_0002_cwef.csv"
    )

    assert first[0] == ["date", "7"]
    assert second[0] == ["date"]

    report = read_json(
        sandbox.output / "export_geometry_report.json"
    )

    assert report["ambiguous_feature_points"] == [
        {
            "point_id": 7,
            "feature_numbers": [1, 2],
        }
    ]


def test_ambiguous_feature_error_policy(
    exporter,
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        exporter,
        "AMBIGUOUS_POINT_POLICY",
        "error",
    )

    feature = single_line_feature()

    write_geojson(
        sandbox.geojson,
        [feature, feature],
    )

    create_database(
        sandbox.database,
        [(7, 10.0, 1000.0, 3)],
    )

    with pytest.raises(
        ValueError,
        match="равные расстояния",
    ):
        exporter.export()


def test_unassigned_points_are_reported(
    exporter,
    sandbox,
):
    write_geojson(
        sandbox.geojson,
        [single_line_feature()],
    )

    create_database(
        sandbox.database,
        [
            (7, 10.0, 1000.0, 3),
            (999, 1000.0, 5000.0, 3),
        ],
    )

    exporter.export()

    rows = read_dict_csv(
        sandbox.output / "unassigned_points.csv"
    )

    assert [
        int(row["point_id"])
        for row in rows
    ] == [999]

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert values[0] == ["date", "7"]


def test_source_database_remains_unchanged(
    exporter,
    sandbox,
):
    prepare_single_line(
        sandbox,
        activity=((7, 1, 30),),
    )

    before = sandbox.database.read_bytes()

    exporter.export()

    after = sandbox.database.read_bytes()

    assert after == before


def test_schema_version_is_checked(
    exporter,
    sandbox,
):
    prepare_single_line(sandbox)

    with sqlite3.connect(sandbox.database) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(
        ValueError,
        match="получена v2",
    ):
        exporter.export()


def test_existing_files_are_not_overwritten_by_default(
    exporter,
    sandbox,
):
    prepare_single_line(sandbox)

    exporter.export()

    with pytest.raises(FileExistsError):
        exporter.export()


def test_explicit_overwrite_is_supported(
    exporter,
    sandbox,
    monkeypatch,
):
    prepare_single_line(
        sandbox,
        activity=((7, 1, 30),),
    )

    exporter.export()

    monkeypatch.setattr(
        exporter,
        "OVERWRITE",
        True,
    )

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert values == [
        ["date", "7"],
        ["2000-02-28", "0"],
        ["2000-02-29", "30"],
        ["2000-03-01", "0"],
    ]


def test_small_sql_and_day_batches_preserve_values(
    exporter,
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(exporter, "DAY_CHUNK_SIZE", 1)
    monkeypatch.setattr(exporter, "SQL_POINT_BATCH_SIZE", 1)

    write_geojson(
        sandbox.geojson,
        [single_line_feature()],
    )

    create_database(
        sandbox.database,
        [
            (90, 5.0, 1000.0, 3),
            (2, 15.0, 1000.0, 3),
            (70, 25.0, 1000.0, 3),
        ],
        (
            (90, 0, 11),
            (2, 1, 22),
            (70, 2, 33),
        ),
    )

    exporter.export()

    assert read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    ) == [
        ["date", "90", "2", "70"],
        ["2000-02-28", "11", "0", "0"],
        ["2000-02-29", "0", "22", "0"],
        ["2000-03-01", "0", "0", "33"],
    ]


def test_empty_feature_still_gets_its_own_files(
    exporter,
    sandbox,
):
    write_geojson(
        sandbox.geojson,
        [
            single_line_feature(),
            make_feature(
                [
                    [
                        (1000.0, 1000.0),
                        (1030.0, 1000.0),
                    ]
                ]
            ),
        ],
    )

    create_database(
        sandbox.database,
        [(7, 10.0, 1000.0, 3)],
    )

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0002_cwef.csv"
    )

    assert values == [
        ["date"],
        ["2000-02-28"],
        ["2000-02-29"],
        ["2000-03-01"],
    ]

    assert read_dict_csv(
        sandbox.output / "coastline_0002_points.csv"
    ) == []

    geojson = read_json(
        sandbox.output / "coastline_0002_points.geojson"
    )

    assert geojson["features"] == []


def test_disconnected_parts_remain_in_one_feature(
    exporter,
    sandbox,
):
    write_geojson(
        sandbox.geojson,
        [
            make_feature(
                [
                    [(0.0, 1000.0), (20.0, 1000.0)],
                    [(100.0, 1000.0), (120.0, 1000.0)],
                ]
            )
        ],
    )

    create_database(
        sandbox.database,
        [
            (7, 10.0, 1000.0, 3),
            (8, 110.0, 1000.0, 3),
        ],
    )

    exporter.export()

    assert len(
        list(sandbox.output.glob("coastline_*_cwef.csv"))
    ) == 1

    report = read_json(
        sandbox.output / "export_geometry_report.json"
    )

    assert report["feature_count"] == 1
    assert report["component_count"] == 2


def test_component_can_be_reversed(
    exporter,
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        exporter,
        "REVERSE_COMPONENTS",
        {(1, 1)},
    )

    write_geojson(
        sandbox.geojson,
        [single_line_feature()],
    )

    create_database(
        sandbox.database,
        [
            (10, 5.0, 1000.0, 3),
            (20, 25.0, 1000.0, 3),
        ],
    )

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert values[0] == ["date", "20", "10"]


def test_closed_component_anchor_changes_start(
    exporter,
    sandbox,
    monkeypatch,
):
    write_geojson(
        sandbox.geojson,
        [
            make_feature(
                [
                    [
                        (0.0, 1000.0),
                        (20.0, 1000.0),
                        (20.0, 1020.0),
                        (0.0, 1020.0),
                        (0.0, 1000.0),
                    ]
                ]
            )
        ],
    )

    monkeypatch.setattr(
        exporter,
        "START_ANCHORS_WGS84",
        {
            (1, 1): wgs84(20.0, 1010.0),
        },
    )

    features = exporter.read_features(sandbox.geojson)
    crs = exporter.metric_crs(features)
    components = exporter.build_components(features, crs)

    assert len(components) == 1

    geometry = components[0].geometry

    assert geometry.is_closed

    assert Point(geometry.coords[0]).distance(
        Point(20.0, 1010.0)
    ) < 0.000001

    assert geometry.length == pytest.approx(80.0)


def test_activity_outside_calendar_is_rejected(
    exporter,
    sandbox,
):
    prepare_single_line(
        sandbox,
        activity=((7, 10, 30),),
    )

    with pytest.raises(
        ValueError,
        match="выходит за календарь",
    ):
        exporter.export()


def test_all_zero_series_has_full_calendar(
    exporter,
    sandbox,
):
    prepare_single_line(sandbox)

    exporter.export()

    values = read_csv(
        sandbox.output / "coastline_0001_cwef.csv"
    )

    assert [
        row[0]
        for row in values[1:]
    ] == EXPECTED_DATES

    assert [
        row[1]
        for row in values[1:]
    ] == ["0", "0", "0"]


def test_unknown_component_in_settings_is_rejected(
    exporter,
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        exporter,
        "REVERSE_COMPONENTS",
        {(1, 99)},
    )

    prepare_single_line(sandbox)

    with pytest.raises(
        ValueError,
        match="Несуществующие компоненты",
    ):
        exporter.export()


def test_branching_geometry_is_rejected(
    exporter,
    sandbox,
):
    write_geojson(
        sandbox.geojson,
        [
            make_feature(
                [
                    [(0.0, 1000.0), (10.0, 1000.0)],
                    [(10.0, 1000.0), (20.0, 1000.0)],
                    [(10.0, 1000.0), (10.0, 1010.0)],
                ]
            )
        ],
    )

    features = exporter.read_features(sandbox.geojson)
    crs = exporter.metric_crs(features)

    with pytest.raises(
        ValueError,
        match="разветвление",
    ):
        exporter.build_components(features, crs)


def test_no_partial_output_after_activity_summary_error(
    exporter,
    sandbox,
):
    # Противоречивая тестовая база:
    # положительная activity есть, summary отсутствует.
    prepare_single_line(
        sandbox,
        n_days=None,
        activity=((7, 1, 30),),
    )

    with pytest.raises(
        ValueError,
        match="Несогласованные activity/summary",
    ):
        exporter.export()

    # Временные файлы не должны попасть в конечный каталог.
    assert sandbox.output.is_dir()
    assert list(sandbox.output.iterdir()) == []