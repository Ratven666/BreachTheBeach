from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import signal
import sqlite3
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.weather_history.archive.repository import (
    CompactWeatherRepository,
    WeatherRecord,
    decode_value,
    encode_day,
)
from src.waves.services.wave_activity_db_service import (
    CoastlineWavePoint,
    WaveActivityBuildStats,
    _WorkerPayload,
    _load_fetches,
    _process_point,
)
from src.waves.storage import (
    WaveActivityRepository,
    WaveActivityRow,
    WaveActivitySummaryRow,
    WavePointRow,
)
from src.waves.storage.wave_activity_repository import round_azimuth_deg


ALGORITHM_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_state(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def input_signature(
    paths: list[Path],
) -> tuple[list[dict], dict[Path, tuple[int, int]]]:
    records = []
    states = {}

    for path in sorted(set(p.resolve() for p in paths)):
        if not path.is_file():
            raise FileNotFoundError(path)

        before = file_state(path)

        logger.info("Checking input: {}", path)

        digest = sha256_file(path)

        if file_state(path) != before:
            raise RuntimeError(
                f"Input changed while hashing: {path}"
            )

        records.append({
            "path": str(path),
            "sha256": digest,
        })
        states[path] = before

    return records, states


def assert_inputs_unchanged(
    states: dict[Path, tuple[int, int]],
) -> None:
    for path, state in states.items():
        if file_state(path) != state:
            raise RuntimeError(
                f"Input changed during calculation: {path}"
            )

        if path.suffix == ".db":
            for suffix in ("-wal", "-journal"):
                sidecar = Path(f"{path}{suffix}")

                if sidecar.exists() and sidecar.stat().st_size:
                    raise RuntimeError(
                        f"Input SQLite is being modified: {sidecar}"
                    )


class BulkWeatherRepository(CompactWeatherRepository):
    """
    Сохраняет исходную IDW-арифметику.

    Вместо SQL-запроса на каждый день и каждый узел
    используется временный буфер соседних узлов одной точки.
    После завершения интерполяции буфер освобождается.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)

        self._records: dict | None = None

        self._con.execute("PRAGMA cache_size = -8192")
        self._con.execute("PRAGMA mmap_size = 0")

    def get_interpolated_timeseries(
        self,
        lat,
        lon,
        start_date=None,
        end_date=None,
    ):
        neighbours = self._neighbourhood_3x3(lat, lon)

        if not neighbours:
            return []

        ids = [point.id for point in neighbours]

        conditions = [
            f"point_id IN ({','.join('?' for _ in ids)})"
        ]
        params = list(ids)

        for value, operator in (
            (start_date, ">="),
            (end_date, "<="),
        ):
            if value is not None:
                if isinstance(value, str):
                    value = date.fromisoformat(value)

                conditions.append(f"day {operator} ?")
                params.append(encode_day(value))

        self._records = {}

        try:
            cursor = self._con.execute(
                "SELECT point_id, day, "
                "wind_speed_max_i, wind_speed_mean_i, "
                "wind_gust_max_i, wind_direction_i "
                "FROM weather_days WHERE "
                + " AND ".join(conditions),
                params,
            )

            for row in cursor:
                self._records[(row[0], row[1])] = tuple(row[2:])

            return super().get_interpolated_timeseries(
                lat,
                lon,
                start_date,
                end_date,
            )
        finally:
            self._records = None

    def _get_day_range(
        self,
        neighbours,
        start_date,
        end_date,
    ):
        if self._records is None:
            return super()._get_day_range(
                neighbours,
                start_date,
                end_date,
            )

        return sorted({
            day
            for _, day in self._records
        })

    def get_record_for_point(
        self,
        point,
        obs_date,
    ):
        if self._records is None:
            return super().get_record_for_point(
                point,
                obs_date,
            )

        if isinstance(obs_date, str):
            obs_date = date.fromisoformat(obs_date)

        row = self._records.get((
            point.id,
            encode_day(obs_date),
        ))

        if row is None:
            return None

        return WeatherRecord(
            point=point,
            obs_date=obs_date,
            wind_speed_max=decode_value(row[0]),
            wind_speed_mean=decode_value(row[1]),
            wind_gust_max=decode_value(row[2]),
            wind_direction=decode_value(row[3]),
        )


@dataclass(slots=True)
class PointResult:
    point: CoastlineWavePoint
    rows: list[tuple[int, int, int, int]]
    summary: WaveActivitySummaryRow | None

    skipped: int = 0
    land: int = 0
    no_bathy: int = 0

    seconds: float = 0.0
    error: str | None = None


_worker_weather = None
_worker_session = None
_worker_options = None


def init_worker(
    weather_path: str,
    coastline_path: str,
    options: dict,
) -> None:
    global _worker_weather
    global _worker_session
    global _worker_options

    # Ctrl+C обрабатывает главный процесс.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Не дублируем сообщения загрузки батиметрии от всех workers.
    logger.remove()

    _worker_weather = BulkWeatherRepository(weather_path)

    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(
            Path(coastline_path).as_uri() + "?mode=ro",
            uri=True,
        ),
    )

    _worker_session = Session(engine)
    _worker_options = options


def process_point(
    point: CoastlineWavePoint,
) -> PointResult:
    started = time.monotonic()
    skipped = 0

    try:
        opts = _worker_options

        fetch_map = _load_fetches(
            _worker_session,
            point.point_id,
        )

        # Не держим долгую read-транзакцию SQLAlchemy.
        _worker_session.rollback()

        records = _worker_weather.get_interpolated_timeseries(
            point.lat,
            point.lon,
            opts["start_date"],
            opts["end_date"],
        )

        days = []
        directions = []
        speeds = []

        for rec in records:
            if (
                rec.obs_date is None
                or rec.wind_speed_max is None
                or rec.wind_direction is None
            ):
                skipped += 1
                continue

            try:
                speed = float(rec.wind_speed_max)
                direction = float(rec.wind_direction)
            except (TypeError, ValueError):
                skipped += 1
                continue

            if (
                not math.isfinite(speed)
                or not math.isfinite(direction)
            ):
                skipped += 1
                continue

            try:
                day = encode_day(rec.obs_date)
            except (
                TypeError,
                AttributeError,
                ValueError,
            ):
                skipped += 1
                continue

            days.append(int(day))
            directions.append(int(round(direction)) % 360)
            speeds.append(max(speed, 0.0))

        del records

        payload = _WorkerPayload(
            point=point,
            fetch_map=fetch_map,
            weather_days=days,
            weather_dirs=directions,
            weather_speeds_kmh=speeds,
            skipped_weather=skipped,
            bathy_manifest_path=opts["bathy_manifest_path"],
            bathy_n_steps=opts["bathy_n_steps"],
            overwater_factor=opts["overwater_factor"],
            breaking_coeff=opts["breaking_coeff"],
            rho_water=opts["rho_water"],
            g=opts["g"],
        )

        # Исходная физическая модель без изменения формул.
        result = _process_point(payload)

        summary = (
            None
            if result.error
            else WaveActivitySummaryRow.from_powers(
                point.point_id,
                result.cwef_all,
            )
        )

        return PointResult(
            point=point,
            rows=result.activity_rows,
            summary=summary,
            skipped=result.skipped_weather,
            land=result.land_sector_days,
            no_bathy=result.missing_bathymetry_days,
            seconds=time.monotonic() - started,
            error=result.error,
        )

    except Exception:
        if _worker_session is not None:
            _worker_session.rollback()

        return PointResult(
            point=point,
            rows=[],
            summary=None,
            skipped=skipped,
            seconds=time.monotonic() - started,
            error=traceback.format_exc(),
        )


class CheckpointRepository(WaveActivityRepository):
    def setup(
        self,
        config: dict,
        points: list[CoastlineWavePoint],
        *,
        overwrite=False,
    ):
        con = self._con

        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }

        fresh = not tables

        if fresh:
            self.initialize()
        else:
            self.assert_schema_version()

            if (
                "wave_build_meta" not in tables
                and not overwrite
            ):
                raise ValueError(
                    "Legacy DB has no checkpoint. "
                    "Use another WAVE_OUTPUT_DB "
                    "or explicitly set WAVE_OVERWRITE=true."
                )

        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = FULL")
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute("PRAGMA cache_size = -16384")
        con.execute("PRAGMA temp_store = FILE")

        con.executescript("""
            CREATE TABLE IF NOT EXISTS wave_build_meta (
                id INTEGER PRIMARY KEY CHECK (id=1),
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS wave_build_progress (
                point_id INTEGER PRIMARY KEY
                    REFERENCES wave_points(id)
                    ON DELETE CASCADE,

                status TEXT NOT NULL CHECK (
                    status IN (
                        'pending',
                        'running',
                        'done',
                        'error'
                    )
                ),

                attempts INTEGER NOT NULL DEFAULT 0,
                n_days INTEGER NOT NULL DEFAULT 0,
                n_rows INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                land INTEGER NOT NULL DEFAULT 0,
                no_bathy INTEGER NOT NULL DEFAULT 0,
                seconds REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                error TEXT
            );
        """)

        encoded = json.dumps(
            config,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

        meta = con.execute(
            "SELECT config_json "
            "FROM wave_build_meta WHERE id=1"
        ).fetchone()

        if (
            meta is not None
            and meta[0] != encoded
            and not overwrite
        ):
            raise ValueError(
                "Input data/parameters/code differ from checkpoint. "
                "Choose another WAVE_OUTPUT_DB "
                "or WAVE_OVERWRITE=true."
            )

        if (
            meta is None
            and not fresh
            and not overwrite
        ):
            raise ValueError(
                "Incomplete checkpoint initialization. "
                "Use WAVE_OVERWRITE=true."
            )

        with con:
            if overwrite:
                con.execute("DELETE FROM wave_build_progress")
                con.execute("DELETE FROM wave_build_meta")

                # FK ON DELETE CASCADE удалит связанные результаты.
                con.execute("DELETE FROM wave_points")

                meta = None

            if meta is None:
                for point in points:
                    self.add_point(
                        WavePointRow(
                            point.point_id,
                            point.lon,
                            point.lat,
                            round_azimuth_deg(
                                point.normal_azimuth_deg
                            ),
                        )
                    )

                con.executemany(
                    "INSERT INTO wave_build_progress "
                    "(point_id,status,updated_at) "
                    "VALUES (?,'pending',?)",
                    (
                        (point.point_id, utc_now())
                        for point in points
                    ),
                )

                con.execute(
                    "INSERT INTO wave_build_meta "
                    "VALUES (1,?,'running',?,NULL)",
                    (encoded, utc_now()),
                )

            count = con.execute(
                "SELECT COUNT(*) FROM wave_build_progress"
            ).fetchone()[0]

            if count != len(points):
                raise ValueError(
                    "Checkpoint point count mismatch"
                )

            # running — задания прерванного запуска.
            # error — точки, завершившиеся ошибкой.
            # done без summary также нельзя считать завершённым.
            con.execute("""
                UPDATE wave_build_progress
                SET status='pending'
                WHERE status IN ('running','error')
                   OR (
                        status='done'
                        AND NOT EXISTS (
                            SELECT 1
                            FROM wave_activity_summary s
                            WHERE s.point_id =
                                wave_build_progress.point_id
                        )
                   )
            """)

        done = {
            row[0]
            for row in con.execute(
                "SELECT point_id "
                "FROM wave_build_progress "
                "WHERE status='done'"
            )
        }

        return [
            point
            for point in points
            if point.point_id not in done
        ]

    def mark_running(
        self,
        points: list[CoastlineWavePoint],
    ) -> None:
        with self._con:
            # Производные индексы недействительны,
            # если начинается/продолжается расчёт активности.
            self._con.execute(
                "DELETE FROM wave_exposure_index"
            )

            self._con.execute(
                "UPDATE wave_build_meta "
                "SET status='running',updated_at=?,error=NULL "
                "WHERE id=1",
                (utc_now(),),
            )

            self._con.executemany(
                "UPDATE wave_build_progress "
                "SET status='running',"
                "attempts=attempts+1,"
                "updated_at=?,error=NULL "
                "WHERE point_id=?",
                (
                    (utc_now(), point.point_id)
                    for point in points
                ),
            )

    def save_result(
        self,
        result: PointResult,
        insert_batch_size: int,
    ) -> None:
        """
        Без commit.

        Данные точки, summary и status='done'
        входят в одну внешнюю batch-транзакцию.
        """
        pid = result.point.point_id

        self._con.execute(
            "DELETE FROM wave_activity WHERE point_id=?",
            (pid,),
        )

        self._con.execute(
            "DELETE FROM wave_activity_summary WHERE point_id=?",
            (pid,),
        )

        if result.error is None:
            if result.summary is None:
                raise ValueError(
                    f"Missing summary for point_id={pid}"
                )

            for start in range(
                0,
                len(result.rows),
                insert_batch_size,
            ):
                self.add_activity(
                    WaveActivityRow(*row)
                    for row in result.rows[
                        start:start + insert_batch_size
                    ]
                )

            self.add_summary(result.summary)

        self._con.execute(
            "UPDATE wave_build_progress "
            "SET status=?,n_days=?,n_rows=?,"
            "skipped=?,land=?,no_bathy=?,"
            "seconds=?,updated_at=?,error=? "
            "WHERE point_id=?",
            (
                "error" if result.error else "done",
                (
                    result.summary.n_days
                    if result.summary
                    else 0
                ),
                (
                    len(result.rows)
                    if not result.error
                    else 0
                ),
                result.skipped,
                result.land,
                result.no_bathy,
                result.seconds,
                utc_now(),
                result.error,
                pid,
            ),
        )

    def rollback(self) -> None:
        self._con.rollback()

    def set_status(
        self,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._con:
            self._con.execute(
                "UPDATE wave_build_meta "
                "SET status=?,updated_at=?,error=? "
                "WHERE id=1",
                (status, utc_now(), error),
            )

    def stats(self) -> WaveActivityBuildStats:
        row = self._con.execute("""
            SELECT
                COUNT(*),
                COALESCE(SUM(n_days),0),
                COALESCE(SUM(n_rows),0),
                COALESCE(SUM(skipped),0),
                COALESCE(SUM(land),0),
                COALESCE(SUM(no_bathy),0),
                COALESCE(SUM(status='error'),0)
            FROM wave_build_progress
        """).fetchone()

        return WaveActivityBuildStats(
            *(int(value) for value in row)
        )

    def checkpoint(self) -> None:
        self._con.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )


def run_incremental(
    repository: CheckpointRepository,
    points: list[CoastlineWavePoint],
    *,
    weather_path: Path,
    coastline_path: Path,
    options: dict,
    input_states: dict[Path, tuple[int, int]],
    max_workers: int,
    pool_batch_points: int = 128,
    commit_batch_size: int = 500,
    commit_points: int = 4,
    commit_seconds: float = 10.0,
) -> WaveActivityBuildStats:
    """
    Маленькие задания, ограниченная очередь,
    один SQLite-writer и восстановление по точкам.
    """
    workers = min(
        max_workers,
        max(1, len(points)),
    )

    context = mp.get_context("spawn")

    buffered_points = 0
    buffered_rows = 0

    last_commit = time.monotonic()
    completed = 0

    def flush(force=False):
        nonlocal buffered_points
        nonlocal buffered_rows
        nonlocal last_commit

        should_commit = (
            force
            or buffered_points >= commit_points
            or buffered_rows >= commit_batch_size
            or time.monotonic() - last_commit >= commit_seconds
        )

        if buffered_points and should_commit:
            assert_inputs_unchanged(input_states)

            repository.commit()

            logger.info(
                "COMMIT: points={}, rows={}, session={}/{}",
                buffered_points,
                buffered_rows,
                completed,
                len(points),
            )

            buffered_points = 0
            buffered_rows = 0
            last_commit = time.monotonic()

    try:
        for offset in range(
            0,
            len(points),
            pool_batch_points,
        ):
            assert_inputs_unchanged(input_states)

            group = points[
                offset:offset + pool_batch_points
            ]

            repository.mark_running(group)

            # Пересоздаём весь пул после ограниченной группы точек.
            # Это ограничивает накопление native-ресурсов и кешей,
            # не используя max_tasks_per_child.
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=init_worker,
                initargs=(
                    str(weather_path),
                    str(coastline_path),
                    options,
                ),
            ) as pool:
                iterator = iter(group)
                pending = {}

                def fill():
                    while len(pending) < 2 * workers:
                        point = next(iterator, None)

                        if point is None:
                            break

                        future = pool.submit(
                            process_point,
                            point,
                        )

                        pending[future] = point

                fill()

                while pending:
                    ready, _ = wait(
                        pending,
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in ready:
                        point = pending.pop(future)

                        # Авария процесса прерывает запуск.
                        # Такие задания не становятся done.
                        result = future.result()

                        repository.save_result(
                            result,
                            commit_batch_size,
                        )

                        buffered_points += 1
                        buffered_rows += len(result.rows)
                        completed += 1

                        if result.error:
                            logger.error(
                                "point_id={}: {}",
                                point.point_id,
                                result.error.strip().splitlines()[-1],
                            )
                        else:
                            logger.info(
                                "[{}/{}] point_id={}: "
                                "days={}, rows={}, time={:.2f}s",
                                completed,
                                len(points),
                                point.point_id,
                                result.summary.n_days,
                                len(result.rows),
                                result.seconds,
                            )

                        del result

                        flush()

                    ready.clear()

                    flush()
                    fill()

            flush(force=True)

        flush(force=True)

        assert_inputs_unchanged(input_states)

        stats = repository.stats()

        repository.set_status(
            "completed"
            if not stats.error_point_count
            else "partial"
        )

        repository.optimize()
        repository.checkpoint()

        return stats

    except BaseException:
        message = traceback.format_exc()

        # Уже подтверждённые batch-транзакции сохраняются.
        # Последняя неподтверждённая пачка будет повторена.
        repository.rollback()

        repository.set_status(
            "interrupted",
            message,
        )

        raise
