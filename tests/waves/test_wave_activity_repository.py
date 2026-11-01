from __future__ import annotations

import sqlite3

import pytest

from src.waves.storage import (
    WaveActivityRepository,
    WaveActivityRow,
    WaveActivitySummaryRow,
    WavePointRow,
)


def test_repository_creates_only_required_tables_and_round_trips(tmp_path):
    path = tmp_path / "waves.db"
    with WaveActivityRepository(path) as repo:
        repo.initialize()
        repo.add_point(WavePointRow(7, 30.0, 60.0, 0.5, 0.5, 45.0))
        assert repo.add_activity([WaveActivityRow(7, 3, 359, 1235)]) == 1
        repo.add_summary(WaveActivitySummaryRow.from_powers(7, [0, 100, 300]))

    con = sqlite3.connect(path)
    tables = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert tables == {"wave_points", "wave_activity", "wave_activity_summary"}
    assert con.execute("SELECT * FROM wave_points").fetchone() == (
        7, 30.0, 60.0, 0.5, 0.5, 45.0
    )
    assert con.execute("SELECT * FROM wave_activity").fetchone() == (
        7, 3, 359, 1235
    )
    summary = con.execute(
        "SELECT n_days, n_active_days, mean_power_wm, max_power_wm, "
        "total_energy_mjm FROM wave_activity_summary"
    ).fetchone()
    assert summary[:4] == (3, 2, pytest.approx(133.33), 300)
    assert summary[4] == pytest.approx(34.6)
    con.close()


def test_activity_rejects_missing_point(tmp_path):
    path = tmp_path / "waves.db"
    with WaveActivityRepository(path) as repo:
        repo.initialize()
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_activity([WaveActivityRow(999, 0, 90, 1)])


def test_empty_summary_contains_no_fabricated_statistics():
    row = WaveActivitySummaryRow.from_powers(1, [])
    assert row.n_days == 0
    assert row.n_active_days == 0
    assert row.mean_power_wm is None
    assert row.max_power_wm is None
    assert row.total_energy_mjm == 0.0
