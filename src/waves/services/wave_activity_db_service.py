from __future__ import annotations

"""Быстрый расчёт дневной волновой активности с мультипроцессингом.

Правила хранения:
  * wave_points.normal_azimuth_deg — целый градус [0, 359];
  * wave_activity.cwef_wm — целое округлённое значение, Вт/м;
    дни с округлённым CWEF = 0 не сохраняются;
  * wave_activity_summary — n_days по всем валидным дням,
    статистики (mean, median, std, перцентили, max, штормовые дни)
    только по дням с CWEF > 0 (см. WaveActivitySummaryRow.from_powers).
"""

import math
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.bathymetry.archive import BathymetryArchive
from src.coastline.storage.models import (
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    WindFetchModel,
)
from src.weather_history.archive.repository import (
    CompactWeatherRepository,
    encode_day,
)
from src.waves.energy import WaveEnergyCalculator
from src.waves.errors import WaveBathymetryError
from src.waves.nearshore import (
    BreakingModel,
    NearshoreWaveTransformer,
    RefractionModel,
)
from src.waves.offshore import SMBWaveGrowthModel
from src.waves.storage import (
    WaveActivityRepository,
    WaveActivityRow,
    WaveActivitySummaryRow,
    WavePointRow,
)
from src.waves.storage.wave_activity_repository import (
    round_azimuth_deg,
    round_cwef_wm,
)


CALM_WIND_KMH = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CoastlineWavePoint:
    point_id: int
    lon: float
    lat: float
    # Точное значение для расчёта; в БД сохраняется округлённым.
    normal_azimuth_deg: float


@dataclass(frozen=True, slots=True)
class WaveActivityBuildStats:
    point_count: int
    processed_day_count: int
    activity_count: int
    skipped_weather_rows: int
    land_sector_rows: int
    missing_bathymetry_rows: int
    error_point_count: int


@dataclass
class _PointResult:
    point: CoastlineWavePoint
    activity_rows: list[tuple[int, int, int, int]]
    # Точные CWEF за все валидные дни, включая нулевые.
    cwef_all: list[float]
    skipped_weather: int
    land_sector_days: int
    missing_bathymetry_days: int
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_wave_points(
    session: Session,
    normal_source_id: int,
) -> list[CoastlineWavePoint]:
    normal_source = session.get(CoastlineNormalSourceModel, normal_source_id)
    if normal_source is None:
        raise ValueError(f"CoastlineNormalSource id={normal_source_id} not found")

    rows = session.execute(
        select(CoastlineNormalModel, CoastlinePointModel)
        .join(CoastlinePointModel, CoastlineNormalModel.point_id == CoastlinePointModel.id)
        .where(CoastlineNormalModel.normal_source_id == normal_source_id)
        .order_by(CoastlinePointModel.seq, CoastlinePointModel.id)
    ).all()

    return [
        CoastlineWavePoint(
            point_id=int(point.id),
            lon=float(point.lon),
            lat=float(point.lat),
            normal_azimuth_deg=float(normal.normal_azimuth_deg) % 360.0,
        )
        for normal, point in rows
    ]


def resolve_normal_source_id(session: Session, normal_source_id: int | None) -> int:
    if normal_source_id is not None:
        if session.get(CoastlineNormalSourceModel, normal_source_id) is None:
            raise ValueError(f"CoastlineNormalSource id={normal_source_id} not found")
        return int(normal_source_id)

    value = session.execute(
        select(CoastlineNormalSourceModel.id)
        .join(CoastlineNormalModel, CoastlineNormalModel.normal_source_id == CoastlineNormalSourceModel.id)
        .join(WindFetchModel, WindFetchModel.point_id == CoastlineNormalModel.point_id)
        .group_by(CoastlineNormalSourceModel.id)
        .order_by(CoastlineNormalSourceModel.id.desc())
        .limit(1)
    ).scalar_one_or_none()

    if value is None:
        raise ValueError("No normal source with wind fetches found.")
    return int(value)


def _load_fetches(session: Session, point_id: int) -> dict[int, float]:
    rows = session.execute(
        select(WindFetchModel)
        .where(WindFetchModel.point_id == point_id)
        .order_by(WindFetchModel.azimuth_deg)
    ).scalars().all()

    if not rows:
        raise ValueError(f"No wind fetches for point_id={point_id}")

    result: dict[int, float] = {}
    for row in rows:
        d = int(round(float(row.azimuth_deg))) % 360
        f = float(row.fetch_length_m)
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(f"Invalid fetch: point_id={point_id}, dir={d}, fetch={f}")
        result[d] = f
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Геометрия
# ─────────────────────────────────────────────────────────────────────────────

def _angular_distance_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


def _is_sea_direction(direction_deg: float, shore_normal_deg: float) -> bool:
    return _angular_distance_deg(direction_deg, shore_normal_deg) <= 90.0


def _nearest_fetch(fetch_map: dict[int, float], direction: int) -> float:
    if direction in fetch_map:
        return fetch_map[direction]
    best = min(
        fetch_map.keys(),
        key=lambda d: _angular_distance_deg(d, direction),
    )
    return fetch_map[best]


# ─────────────────────────────────────────────────────────────────────────────
# Worker payload (pickleable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _WorkerPayload:
    point: CoastlineWavePoint
    fetch_map: dict[int, float]
    weather_days: list[int]
    weather_dirs: list[int]
    weather_speeds_kmh: list[float]
    skipped_weather: int
    bathy_manifest_path: str
    bathy_n_steps: int
    overwater_factor: float
    breaking_coeff: float
    rho_water: float
    g: float


# ─────────────────────────────────────────────────────────────────────────────
# Worker (subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _process_point(payload: _WorkerPayload) -> _PointResult:
    point = payload.point

    try:
        smb = SMBWaveGrowthModel(g=payload.g)
        energy = WaveEnergyCalculator(rho_water=payload.rho_water, g=payload.g)

        bathy_archive = BathymetryArchive(Path(payload.bathy_manifest_path))
        bathy_service = bathy_archive.for_point_id(
            point.point_id, n_steps=payload.bathy_n_steps,
        )

        shore_normal = point.normal_azimuth_deg

        transformer = NearshoreWaveTransformer(
            shore_normal_deg=shore_normal,
            profile_provider=bathy_service,
            breaking_model=BreakingModel(gamma_b=payload.breaking_coeff),
            refraction_model=RefractionModel(g=payload.g),
        )

        # Прогрев кэша батиметрии: по одному разу на уникальное морское направление.
        sea_dirs = {
            d
            for d, spd in zip(payload.weather_dirs, payload.weather_speeds_kmh)
            if spd >= CALM_WIND_KMH and _is_sea_direction(d, shore_normal)
        }
        for d in sorted(sea_dirs):
            try:
                transformer.transform(d, 1.0, 5.0)
            except WaveBathymetryError:
                pass

        activity_rows: list[tuple[int, int, int, int]] = []
        cwef_all: list[float] = []
        land_days = 0
        missing_bathy = 0

        for day, direction, wind_kmh in zip(
            payload.weather_days,
            payload.weather_dirs,
            payload.weather_speeds_kmh,
        ):
            cwef_wm = 0.0

            if not _is_sea_direction(direction, shore_normal):
                land_days += 1
            elif wind_kmh >= CALM_WIND_KMH:
                wind_ms = wind_kmh / 3.6 * payload.overwater_factor
                fetch_m = _nearest_fetch(payload.fetch_map, direction)
                hs_off, tp_s = smb.calculate(wind_ms, fetch_m)

                try:
                    ns = transformer.transform(direction, hs_off, tp_s)
                except WaveBathymetryError:
                    missing_bathy += 1
                else:
                    cwef_wm = energy.cwef(
                        energy.wave_power(ns.hs_nearshore_m, tp_s),
                        ns.cos_shore,
                    )
                    if not math.isfinite(cwef_wm) or cwef_wm < 0.0:
                        cwef_wm = 0.0

            # Полный ряд (с нулями) нужен для n_days; статистики по CWEF > 0
            # отбираются внутри WaveActivitySummaryRow.from_powers.
            cwef_all.append(float(cwef_wm))

            # В wave_activity — целое округлённое значение; нули не сохраняются.
            cwef_int = round_cwef_wm(cwef_wm)
            if cwef_int >= 1:
                activity_rows.append((
                    point.point_id,
                    int(day),
                    int(direction) % 360,
                    cwef_int,
                ))

        return _PointResult(
            point=point,
            activity_rows=activity_rows,
            cwef_all=cwef_all,
            skipped_weather=payload.skipped_weather,
            land_sector_days=land_days,
            missing_bathymetry_days=missing_bathy,
        )

    except Exception:
        return _PointResult(
            point=point,
            activity_rows=[],
            cwef_all=[],
            skipped_weather=payload.skipped_weather,
            land_sector_days=0,
            missing_bathymetry_days=0,
            error=traceback.format_exc(),
        )


def _error_result(payload: _WorkerPayload, exc: BaseException) -> _PointResult:
    return _PointResult(
        point=payload.point,
        activity_rows=[],
        cwef_all=[],
        skipped_weather=payload.skipped_weather,
        land_sector_days=0,
        missing_bathymetry_days=0,
        error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Главный строитель
# ─────────────────────────────────────────────────────────────────────────────

class WaveActivityDatabaseBuilder:
    def __init__(
        self,
        coastline_session: Session,
        weather_repository: CompactWeatherRepository,
        bathymetry_archive: BathymetryArchive,
        output_repository: WaveActivityRepository,
        *,
        normal_source_id: int,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        overwater_factor: float = 1.1,
        breaking_coeff: float = 0.55,
        rho_water: float = 1025.0,
        g: float = 9.81,
        bathy_n_steps: int = 200,
        commit_batch_size: int = 500,
        max_workers: int | None = None,
    ) -> None:
        if overwater_factor <= 0.0:
            raise ValueError("overwater_factor must be > 0")
        if not (0.0 < breaking_coeff <= 1.0):
            raise ValueError("breaking_coeff must be in (0, 1]")
        if rho_water <= 0.0:
            raise ValueError("rho_water must be > 0")
        if g <= 0.0:
            raise ValueError("g must be > 0")
        if bathy_n_steps < 2:
            raise ValueError("bathy_n_steps must be >= 2")
        if commit_batch_size < 1:
            raise ValueError("commit_batch_size must be >= 1")

        self._session = coastline_session
        self._weather = weather_repository
        self._bathymetry = bathymetry_archive
        self._output = output_repository
        self._normal_source_id = int(normal_source_id)
        self._start_date = start_date
        self._end_date = end_date
        self._overwater_factor = float(overwater_factor)
        self._breaking_coeff = float(breaking_coeff)
        self._rho_water = float(rho_water)
        self._g = float(g)
        self._bathy_n_steps = int(bathy_n_steps)
        self._commit_batch_size = int(commit_batch_size)
        self._max_workers = max_workers

    # ── Подготовка ──────────────────────────────────────────────────────────

    def _prepare_weather(
        self,
        point: CoastlineWavePoint,
    ) -> tuple[list[int], list[int], list[float], int]:
        weather_records = self._weather.get_interpolated_timeseries(
            point.lat, point.lon,
            self._start_date, self._end_date,
        )

        days: list[int] = []
        dirs: list[int] = []
        speeds: list[float] = []
        skipped = 0

        for rec in weather_records:
            if (
                rec.obs_date is None
                or rec.wind_speed_max is None
                or rec.wind_direction is None
            ):
                skipped += 1
                continue
            try:
                spd = float(rec.wind_speed_max)
                d = float(rec.wind_direction)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not math.isfinite(spd) or not math.isfinite(d):
                skipped += 1
                continue
            try:
                encoded = encode_day(rec.obs_date)
            except (TypeError, AttributeError, ValueError):
                skipped += 1
                continue

            days.append(int(encoded))
            dirs.append(int(round(d)) % 360)
            speeds.append(max(spd, 0.0))

        return days, dirs, speeds, skipped

    def _build_payloads(self, points: list[CoastlineWavePoint]) -> list[_WorkerPayload]:
        logger.info("Preparing payloads for {} points...", len(points))
        payloads: list[_WorkerPayload] = []
        bathy_manifest = str(self._bathymetry.manifest_path)

        for i, point in enumerate(points, start=1):
            fetch_map = _load_fetches(self._session, point.point_id)
            days, dirs, speeds, skipped = self._prepare_weather(point)

            payloads.append(_WorkerPayload(
                point=point,
                fetch_map=fetch_map,
                weather_days=days,
                weather_dirs=dirs,
                weather_speeds_kmh=speeds,
                skipped_weather=skipped,
                bathy_manifest_path=bathy_manifest,
                bathy_n_steps=self._bathy_n_steps,
                overwater_factor=self._overwater_factor,
                breaking_coeff=self._breaking_coeff,
                rho_water=self._rho_water,
                g=self._g,
            ))

            if i % 50 == 0 or i == len(points):
                logger.info("  Payloads prepared: {}/{}", i, len(points))

        return payloads

    # ── Расчёт ──────────────────────────────────────────────────────────────

    def _compute(self, payloads: list[_WorkerPayload]) -> dict[int, _PointResult]:
        n_workers = min(self._resolve_workers(), max(len(payloads), 1))
        total = len(payloads)

        logger.info(
            "Wave activity calculation started: points={}, normal_source_id={}",
            total, self._normal_source_id,
        )
        logger.info("Using {} worker processes", n_workers)

        results: dict[int, _PointResult] = {}

        if n_workers == 1:
            for done, payload in enumerate(payloads, start=1):
                result = _process_point(payload)
                results[result.point.point_id] = result
                if done % 20 == 0 or done == total:
                    logger.info("  Computed: {}/{}", done, total)
            return results

        # spawn вместо fork — SQLite-соединения не наследуются дочерними процессами.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_process_point, p): p for p in payloads}
            done = 0
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # падение процесса / ошибка сериализации
                    result = _error_result(payload, exc)
                results[payload.point.point_id] = result
                done += 1
                if done % 20 == 0 or done == total:
                    logger.info("  Computed: {}/{}", done, total)

        return results

    # ── Главный метод ───────────────────────────────────────────────────────

    def run(self) -> WaveActivityBuildStats:
        points = load_wave_points(self._session, self._normal_source_id)
        if not points:
            raise ValueError(f"Normal source id={self._normal_source_id} contains no points")

        if self._bathymetry.point_source_id != self._point_source_id():
            raise ValueError(
                "Bathymetry archive and normal source refer to different point sources"
            )

        self._output.initialize()

        payloads = self._build_payloads(points)
        results = self._compute(payloads)

        logger.info("Writing results to database...")

        total = len(points)
        total_activity = 0
        total_processed_days = 0
        skipped_weather = 0
        land_sector_rows = 0
        missing_bathymetry = 0
        errors = 0

        for idx, point in enumerate(points, start=1):
            result = results.get(point.point_id)

            self._output.add_point(WavePointRow(
                id=point.point_id,
                lon=point.lon,
                lat=point.lat,
                normal_azimuth_deg=round_azimuth_deg(point.normal_azimuth_deg),
            ))

            if result is None or result.error is not None:
                errors += 1
                message = result.error if result is not None else "no result"
                logger.error(
                    "[{}/{}] point_id={}: ERROR: {}",
                    idx, total, point.point_id,
                    message.strip().splitlines()[-1] if message.strip() else "unknown error",
                )
                if result is not None:
                    skipped_weather += result.skipped_weather
                self._output.add_summary(
                    WaveActivitySummaryRow.from_powers(point.point_id, [])
                )
                self._output.commit()
                continue

            batch = [
                WaveActivityRow(
                    point_id=r[0],
                    day=r[1],
                    wind_azimuth_deg=r[2],
                    cwef_wm=r[3],
                )
                for r in result.activity_rows
            ]
            stored_rows = 0
            for start in range(0, len(batch), self._commit_batch_size):
                stored_rows += self._output.add_activity(
                    batch[start: start + self._commit_batch_size]
                )
                self._output.commit()
            total_activity += stored_rows

            # Статистики — только по дням с CWEF > 0; n_days — по всем дням.
            summary = WaveActivitySummaryRow.from_powers(
                point.point_id, result.cwef_all,
            )
            self._output.add_summary(summary)
            self._output.commit()

            total_processed_days += summary.n_days
            skipped_weather += result.skipped_weather
            land_sector_rows += result.land_sector_days
            missing_bathymetry += result.missing_bathymetry_days

            # Дни с 0 < CWEF < 0.5 входят в статистику, но не в wave_activity.
            below_rounding = summary.n_active_days - stored_rows

            logger.info(
                "[{}/{}] point_id={}: days={}, active_days={}, stored_rows={}, "
                "below_1wm={}, mean={}, median={}, land={}, no_bathy={}, skip_wx={}",
                idx, total, point.point_id,
                summary.n_days,
                summary.n_active_days,
                stored_rows,
                below_rounding,
                summary.mean_cwef_wm,
                summary.median_cwef_wm,
                result.land_sector_days,
                result.missing_bathymetry_days,
                result.skipped_weather,
            )

        self._output.optimize()

        logger.success(
            "Wave activity database ready: "
            "points={}, processed_days={}, active_rows={}, "
            "skipped_weather={}, land_sector_days={}, "
            "missing_bathymetry_days={}, calc_errors={}",
            total, total_processed_days, total_activity,
            skipped_weather, land_sector_rows, missing_bathymetry, errors,
        )

        return WaveActivityBuildStats(
            point_count=total,
            processed_day_count=total_processed_days,
            activity_count=total_activity,
            skipped_weather_rows=skipped_weather,
            land_sector_rows=land_sector_rows,
            missing_bathymetry_rows=missing_bathymetry,
            error_point_count=errors,
        )

    def _resolve_workers(self) -> int:
        if self._max_workers is not None:
            return max(1, int(self._max_workers))
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)

    def _point_source_id(self) -> int:
        return int(
            self._session.execute(
                select(CoastlineNormalSourceModel.point_source_id)
                .where(CoastlineNormalSourceModel.id == self._normal_source_id)
            ).scalar_one()
        )
