from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTER_FILE = PROJECT_ROOT / 'db_pypeline' / '5_8_export_cwef_by_coastline.py'
spec = importlib.util.spec_from_file_location('cwef_exporter_under_test', EXPORTER_FILE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

# Не полноценные наблюдения: контролируемые fixtures только для unit-тестов.
# DDL взят из исходников schema.py ветки feature/db_feature, schema v3.
schema_text = (PROJECT_ROOT / 'src' / 'waves' / 'storage' / 'schema.py').read_text()
SCHEMA_SQL = schema_text.split('SCHEMA_SQL = f"""', 1)[1].rsplit('"""', 1)[0]
SCHEMA_SQL = SCHEMA_SQL.replace('{MAX_DAY_OFFSET}', '65535')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.reader(stream))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'COASTLINES_GEOJSON', tmp_path / 'coasts.geojson')
    monkeypatch.setattr(m, 'WAVE_ACTIVITY_DB', tmp_path / 'wave_activity.db')
    monkeypatch.setattr(m, 'OUTPUT_DIR', tmp_path / 'csv')
    monkeypatch.setattr(m, 'START_DATE', date(2000, 2, 28))
    monkeypatch.setattr(m, 'END_DATE', date(2000, 3, 1))
    monkeypatch.setattr(m, 'WORKING_CRS', 'EPSG:32634')
    monkeypatch.setattr(m, 'REVERSE_LINE_NUMBERS', set())
    monkeypatch.setattr(m, 'MAX_DISTANCE_TO_LINE_M', 5.)
    monkeypatch.setattr(m, 'AMBIGUOUS_POINT_POLICY', 'all')
    monkeypatch.setattr(m, 'MISSING_DAY_POLICY', 'strict')
    monkeypatch.setattr(m, 'OVERWRITE', False)
    return monkeypatch


def setup_case(lines, points, activity=(), n_days=None, multi=False, input_crs=None):
    geometries = ([{'type': 'MultiLineString', 'coordinates': lines}] if multi else
                  [{'type': 'LineString', 'coordinates': line} for line in lines])
    obj = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {}, 'geometry': g} for g in geometries]}
    if input_crs:
        obj['crs'] = {'type': 'name', 'properties': {'name': input_crs}}
    m.COASTLINES_GEOJSON.write_text(json.dumps(obj))
    with sqlite3.connect(m.WAVE_ACTIVITY_DB) as con:
        con.executescript(SCHEMA_SQL)
        con.execute('PRAGMA user_version=3')
        con.executemany('INSERT INTO wave_points VALUES (?, ?, ?, ?)',
                        [(pid, lon, lat, 270) for pid, lon, lat in points])
        for pid, _, _ in points:
            n = (n_days or {}).get(pid, 3)
            if n is not None:
                con.execute('INSERT INTO wave_activity_summary '
                            '(point_id,n_days,n_active_days,n_storm_days_p90,total_energy_mjm) '
                            'VALUES (?,?,0,0,0)', (pid, n))
        first = (m.START_DATE - m.EPOCH).days
        con.executemany('INSERT INTO wave_activity VALUES (?, ?, 270, ?)',
                        [(pid, first + offset, value) for pid, offset, value in activity])


LINE = [[20.0, 55.0], [20.01, 55.0]]
POINTS = [(90, 20.0, 55.0), (2, 20.005, 55.0), (70, 20.01, 55.0)]


def test_ids_chainage_dates_sparse_zero_and_readonly(config):
    setup_case([LINE], POINTS, [(90, 0, 125), (2, 1, 501), (70, 2, 999)])
    before = m.WAVE_ACTIVITY_DB.read_bytes()
    paths = m.export()
    assert len(paths) == 2
    values = read_csv(paths[0])
    assert values == [['date', '90', '2', '70'], ['2000-02-28', '125', '0', '0'],
                      ['2000-02-29', '0', '501', '0'], ['2000-03-01', '0', '0', '999']]
    coords = read_csv(paths[1])
    assert [row[0] for row in coords[1:]] == ['90', '2', '70']
    chainages = [float(row[2]) for row in coords[1:]]
    assert chainages == sorted(chainages) and chainages[0] == 0
    assert m.WAVE_ACTIVITY_DB.read_bytes() == before


def test_reversal(config):
    setup_case([LINE], POINTS)
    config.setattr(m, 'REVERSE_LINE_NUMBERS', {1})
    values = read_csv(m.export()[0])
    assert values[0] == ['date', '70', '2', '90']


def test_multiline_input_order_and_empty_line(config):
    other = [[20., 55.01], [20.01, 55.01]]
    setup_case([other, LINE], POINTS, multi=True)
    paths = m.export()
    assert len(paths) == 4
    assert paths[0].name == 'coastline_0001_cwef.csv'
    assert read_csv(paths[0]) == [['date'], ['2000-02-28'], ['2000-02-29'], ['2000-03-01']]
    assert read_csv(paths[2])[0] == ['date', '90', '2', '70']
    assert read_csv(paths[3])[1][8:10] == ['1', '2']


def test_shared_end_point_all(config):
    second = [[20.01, 55.], [20.02, 55.]]
    setup_case([LINE, second], [(70, 20.01, 55.)])
    paths = m.export()
    assert read_csv(paths[0])[0] == ['date', '70']
    assert read_csv(paths[2])[0] == ['date', '70']


def test_ambiguous_error(config):
    setup_case([LINE, LINE], POINTS)
    config.setattr(m, 'AMBIGUOUS_POINT_POLICY', 'error')
    with pytest.raises(ValueError, match='одинаково близкие'):
        m.export()
    assert not m.OUTPUT_DIR.exists()


def test_ambiguous_first(config):
    setup_case([LINE, LINE], POINTS)
    config.setattr(m, 'AMBIGUOUS_POINT_POLICY', 'first')
    paths = m.export()
    assert read_csv(paths[2])[0] == ['date']


def test_partial_coverage_strict(config):
    setup_case([LINE], POINTS, n_days={2: 2})
    with pytest.raises(ValueError, match='не хранит даты'):
        m.export()
    assert not m.OUTPUT_DIR.exists()


def test_partial_nan_and_failed_points(config):
    setup_case([LINE], POINTS, [(2, 1, 8)], n_days={90: 0, 2: 2, 70: None})
    config.setattr(m, 'MISSING_DAY_POLICY', 'nan')
    rows = read_csv(m.export()[0])
    assert rows[1:] == [['2000-02-28', '', '', ''], ['2000-02-29', '', '8', ''],
                       ['2000-03-01', '', '', '']]


def test_force_zero_does_not_forge_failed_points(config):
    setup_case([LINE], POINTS, [(2, 0, 5)], n_days={90: 0, 2: 2, 70: None})
    config.setattr(m, 'MISSING_DAY_POLICY', 'zero')
    rows = read_csv(m.export()[0])
    assert rows[1:] == [['2000-02-28', '', '5', ''], ['2000-02-29', '', '0', ''],
                       ['2000-03-01', '', '0', '']]


def test_no_active_rows_still_full_calendar(config):
    setup_case([LINE], POINTS)
    config.setattr(m, 'END_DATE', None)
    rows = read_csv(m.export()[0])
    assert len(rows) == 4
    assert all(row[1:] == ['0', '0', '0'] for row in rows[1:])


def test_chunk_and_sql_batches(config):
    setup_case([LINE], POINTS, [(90, 0, 2), (2, 1, 3), (70, 2, 4)])
    config.setattr(m, 'DAY_CHUNK_SIZE', 1)
    config.setattr(m, 'SQL_POINT_BATCH_SIZE', 1)
    rows = read_csv(m.export()[0])
    assert rows[1:] == [['2000-02-28', '2', '0', '0'], ['2000-02-29', '0', '3', '0'],
                       ['2000-03-01', '0', '0', '4']]


def test_projected_geojson(config):
    transformer = Transformer.from_crs(4326, 32634, always_xy=True)
    projected = [list(transformer.transform(*xy)) for xy in LINE]
    setup_case([projected], POINTS, input_crs='EPSG:32634')
    assert read_csv(m.export()[0])[0] == ['date', '90', '2', '70']


def test_outside_tolerance_excluded(config):
    setup_case([LINE], [*POINTS, (444, 20., 55.02)])
    assert '444' not in read_csv(m.export()[0])[0]


def test_refuses_all_unmatched(config):
    setup_case([LINE], [(444, 22., 58.)])
    with pytest.raises(ValueError, match='Ни одна точка'):
        m.export()


def test_schema_rejected(config):
    setup_case([LINE], POINTS)
    with sqlite3.connect(m.WAVE_ACTIVITY_DB) as con:
        con.execute('PRAGMA user_version=2')
    with pytest.raises(ValueError, match='v2'):
        m.export()


def test_overwrite_guard(config):
    setup_case([LINE], POINTS)
    m.export()
    with pytest.raises(FileExistsError):
        m.export()


def test_dates_must_cover_active_rows(config):
    setup_case([LINE], POINTS, [(2, 5, 10)])
    with pytest.raises(ValueError, match='выходят за период'):
        m.export()


def test_staging_no_files_after_failure(config):
    setup_case([LINE], POINTS, [(2, 1, 10)], n_days={2: None})
    with pytest.raises(ValueError, match='отсутствует summary'):
        m.export()
    assert list(m.OUTPUT_DIR.iterdir()) == []


def test_automatic_utm(config):
    setup_case([LINE], POINTS)
    config.setattr(m, 'WORKING_CRS', None)
    assert read_csv(m.export()[0])[0] == ['date', '90', '2', '70']


def test_self_intersection_rejected(config):
    crossed = [[20.,55.], [20.01,55.01], [20.,55.01], [20.01,55.]]
    setup_case([crossed], POINTS)
    with pytest.raises(ValueError, match='самопересечение'):
        m.export()


def test_pandas_dataframe_roundtrip(config):
    import pandas as pd
    setup_case([LINE], POINTS, [(2, 1, 42)], n_days={70: 0})
    values_path, coords_path = m.export()
    frame = pd.read_csv(values_path, index_col='date', parse_dates=['date']).astype('Int64')
    coords = pd.read_csv(coords_path).set_index('point_id')
    frame.columns = frame.columns.astype('int64')
    coords = coords.loc[frame.columns]
    assert frame.index[1] == pd.Timestamp('2000-02-29')
    assert frame.loc['2000-02-29', 2] == 42
    assert frame[70].isna().all()
    assert list(frame.columns) == list(coords.index) == [90, 2, 70]


def test_duplicate_coordinates_different_ids_preserved(config):
    setup_case([LINE], [(99, 20., 55.), (3, 20., 55.)])
    assert read_csv(m.export()[0])[0] == ['date', '3', '99']


def test_closed_coastline_allowed(config):
    ring = [[20.,55.], [20.01,55.], [20.01,55.01], [20.,55.01], [20.,55.]]
    setup_case([ring], POINTS)
    assert read_csv(m.export()[0])[0] == ['date', '90', '2', '70']


def test_geographic_working_crs_rejected(config):
    setup_case([LINE], POINTS)
    config.setattr(m, 'WORKING_CRS', 'EPSG:4326')
    with pytest.raises(ValueError, match='проекцией'):
        m.export()


def test_nan_policy_still_recognizes_completely_valid_all_zero_point(config):
    setup_case([LINE], POINTS, n_days={90: 3, 2: 2, 70: 0})
    config.setattr(m, 'MISSING_DAY_POLICY', 'nan')
    rows = read_csv(m.export()[0])
    assert all(row[1:] == ['0', '', ''] for row in rows[1:])
