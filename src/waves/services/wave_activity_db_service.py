from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd
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
from src.waves.fetch import FetchLookup
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


@dataclass(frozen=True, slots=True)
class CoastlineWavePoint:
    point_id: int
    lon: float
    lat: float
    normal_x: float
    normal_y: float
    normal_azimuth_deg: float


@dataclass(frozen=True, slots=True)
class WaveActivityBuildStats:
    point_count: int
    processed_day_count: int
    activity_count: int
    skipped_weather_rows: int
    land_sector_rows: int
    missing_bathymetry_rows: int


def load_wave_points(
    session: Session,
    normal_source_id: int,
) -> list[CoastlineWavePoint]:
    normal_source = session.get(
        CoastlineNormalSourceModel,
        normal_source_id,
    )

    if normal_source is None:
        raise ValueError(
            f"CoastlineNormalSource id={normal_source_id} not found"
        )

    rows = session.execute(
        select(
            CoastlineNormalModel,
            CoastlinePointModel,
        )
        .join(
            CoastlinePointModel,
            CoastlineNormalModel.point_id
            == CoastlinePointModel.id,
        )
        .where(
            CoastlineNormalModel.normal_source_id
            == normal_source_id
        )
        .order_by(
            CoastlinePointModel.seq,
            CoastlinePointModel.id,
        )
    ).all()

    return [
        CoastlineWavePoint(
            point_id=int(point.id),
            lon=float(point.lon),
            lat=float(point.lat),
            normal_x=float(normal.nx),
            normal_y=float(normal.ny),
            normal_azimuth_deg=(
                float(normal.normal_azimuth_deg) % 360.0
            ),
        )
        for normal, point in rows
    ]


def resolve_normal_source_id(
    session: Session,
    normal_source_id: int | None,
) -> int:
    if normal_source_id is not None:
        source = session.get(
            CoastlineNormalSourceModel,
            normal_source_id,
        )

        if source is None:
            raise ValueError(
                f"CoastlineNormalSource "
                f"id={normal_source_id} not found"
            )

        return int(normal_source_id)

    value = session.execute(
        select(CoastlineNormalSourceModel.id)
        .join(
            CoastlineNormalModel,
            CoastlineNormalModel.normal_source_id
            == CoastlineNormalSourceModel.id,
        )
        .join(
            WindFetchModel,
            WindFetchModel.point_id
            == CoastlineNormalModel.point_id,
        )
        .group_by(CoastlineNormalSourceModel.id)
        .order_by(CoastlineNormalSourceModel.id.desc())
        .limit(1)
    ).scalar_one_or_none()

    if value is None:
        raise ValueError(
            "No normal source with wind fetches found. "
            "Run database pipeline steps 1-3 first."
        )

    return int(value)


def _angular_distance_deg(
    first_deg: float,
    second_deg: float,
) -> float:
    difference = abs(
        (float(first_deg) - float(second_deg)) % 360.0
    )
    return (
        difference
        if difference <= 180.0
        else 360.0 - difference
    )


def _is_sea_direction(
    direction_deg: float,
    shore_normal_deg: float,
) -> bool:
    return (
        _angular_distance_deg(
            direction_deg,
            shore_normal_deg,
        )
        <= 90.0
    )


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
        commit_batch_size: int = 5_000,
    ) -> None:
        if overwater_factor <= 0.0:
            raise ValueError(
                "overwater_factor must be > 0"
            )

        if not (0.0 < breaking_coeff <= 1.0):
            raise ValueError(
                "breaking_coeff must be in (0, 1]"
            )

        if rho_water <= 0.0:
            raise ValueError(
                "rho_water must be > 0"
            )

        if g <= 0.0:
            raise ValueError(
                "g must be > 0"
            )

        if bathy_n_steps < 2:
            raise ValueError(
                "bathy_n_steps must be >= 2"
            )

        if commit_batch_size < 1:
            raise ValueError(
                "commit_batch_size must be >= 1"
            )

        self._session = coastline_session
        self._weather = weather_repository
        self._bathymetry = bathymetry_archive
        self._output = output_repository

        self._normal_source_id = int(normal_source_id)
        self._start_date = start_date
        self._end_date = end_date

        self._overwater_factor = float(
            overwater_factor
        )
        self._breaking_coeff = float(
            breaking_coeff
        )
        self._rho_water = float(rho_water)
        self._g = float(g)

        self._bathy_n_steps = int(
            bathy_n_steps
        )
        self._commit_batch_size = int(
            commit_batch_size
        )

        self._smb = SMBWaveGrowthModel(
            g=self._g
        )
        self._energy = WaveEnergyCalculator(
            rho_water=self._rho_water,
            g=self._g,
        )

    def run(self) -> WaveActivityBuildStats:
        points = load_wave_points(
            self._session,
            self._normal_source_id,
        )

        if not points:
            raise ValueError(
                f"Normal source id={self._normal_source_id} "
                "contains no points"
            )

        if (
            self._bathymetry.point_source_id
            != self._point_source_id()
        ):
            raise ValueError(
                "Bathymetry archive and selected normal source "
                "refer to different coastline point sources"
            )

        self._output.initialize()

        total_activity = 0
        total_processed_days = 0
        skipped_weather = 0
        land_sector_rows = 0
        missing_bathymetry = 0

        total_points = len(points)

        logger.info(
            "Wave activity calculation started: "
            "points={}, normal_source_id={}",
            total_points,
            self._normal_source_id,
        )

        for index, point in enumerate(
            points,
            start=1,
        ):
            logger.info(
                "[{}/{}] point_id={}: "
                "loading fetches and weather",
                index,
                total_points,
                point.point_id,
            )

            self._output.add_point(
                WavePointRow(
                    id=point.point_id,
                    lon=point.lon,
                    lat=point.lat,
                    normal_x=point.normal_x,
                    normal_y=point.normal_y,
                    normal_azimuth_deg=(
                        point.normal_azimuth_deg
                    ),
                )
            )

            fetch_lookup = self._fetch_lookup(
                point.point_id
            )

            bathy_service = (
                self._bathymetry.for_point_id(
                    point.point_id,
                    n_steps=self._bathy_n_steps,
                )
            )

            transformer = NearshoreWaveTransformer(
                shore_normal_deg=(
                    point.normal_azimuth_deg
                ),
                profile_provider=bathy_service,
                breaking_model=BreakingModel(
                    gamma_b=self._breaking_coeff
                ),
                refraction_model=RefractionModel(
                    g=self._g
                ),
            )

            weather = (
                self._weather
                .get_interpolated_timeseries(
                    point.lat,
                    point.lon,
                    self._start_date,
                    self._end_date,
                )
            )

            pending: list[WaveActivityRow] = []
            cwef_values: list[float] = []

            point_land_days = 0
            point_missing_bathy = 0
            point_skipped_weather = 0

            for record in weather:
                prepared = self._prepare_weather_record(
                    record.wind_speed_max,
                    record.wind_direction,
                )

                if prepared is None:
                    skipped_weather += 1
                    point_skipped_weather += 1
                    continue

                wind_kmh, direction = prepared
                total_processed_days += 1

                cwef_wm = 0.0

                # Fetch хранится только в морском секторе ±90°
                # относительно нормали, направленной в море.
                # Ветер из сушевого сектора является валидным
                # метеорологическим днём, но его береговая
                # волновая экспозиция равна нулю.
                if not _is_sea_direction(
                    direction,
                    point.normal_azimuth_deg,
                ):
                    land_sector_rows += 1
                    point_land_days += 1

                elif wind_kmh >= 0.1:
                    wind_ms = (
                        wind_kmh
                        / 3.6
                        * self._overwater_factor
                    )

                    fetch_m = fetch_lookup.get_fetch(
                        direction
                    )

                    hs_offshore, tp_s = (
                        self._smb.calculate(
                            wind_ms,
                            fetch_m,
                        )
                    )

                    try:
                        nearshore = transformer.transform(
                            direction,
                            hs_offshore,
                            tp_s,
                        )
                    except WaveBathymetryError as exc:
                        # Не создаём искусственное волновое
                        # воздействие из условных глубин.
                        # Отсутствие положительных глубин означает,
                        # что для этого луча морской профиль
                        # подтвердить нельзя.
                        missing_bathymetry += 1
                        point_missing_bathy += 1

                        logger.debug(
                            "point_id={}, direction={}°: "
                            "CWEF=0 because bathymetry profile "
                            "is invalid: {}",
                            point.point_id,
                            direction,
                            exc,
                        )
                    else:
                        wave_power_wm = (
                            self._energy.wave_power(
                                nearshore.hs_nearshore_m,
                                tp_s,
                            )
                        )

                        cwef_wm = self._energy.cwef(
                            wave_power_wm,
                            nearshore.cos_shore,
                        )

                        if (
                            not math.isfinite(cwef_wm)
                            or cwef_wm < 0.0
                        ):
                            raise ValueError(
                                "Invalid CWEF: "
                                f"point_id={point.point_id}, "
                                f"date={record.obs_date}, "
                                f"direction={direction}, "
                                f"value={cwef_wm}"
                            )

                cwef_values.append(
                    float(cwef_wm)
                )

                rounded_cwef = round(float(cwef_wm), 6)
                if rounded_cwef > 0.0:
                    pending.append(
                        WaveActivityRow(
                            point_id=point.point_id,
                            day=encode_day(record.obs_date),
                            wind_azimuth_deg=direction,
                            cwef_wm=rounded_cwef,
                        )
                    )

                if (
                    len(pending)
                    >= self._commit_batch_size
                ):
                    total_activity += (
                        self._output.add_activity(
                            pending
                        )
                    )
                    pending.clear()
                    self._output.commit()

            total_activity += (
                self._output.add_activity(
                    pending
                )
            )

            summary = (
                WaveActivitySummaryRow.from_powers(
                    point.point_id,
                    cwef_values,
                )
            )
            self._output.add_summary(summary)
            self._output.commit()

            logger.info(
                "[{}/{}] point_id={}: "
                "days={}, active_days={}, "
                "land_sector_days={}, "
                "missing_bathymetry_days={}, "
                "skipped_weather={}",
                index,
                total_points,
                point.point_id,
                summary.n_days,
                summary.n_active_days,
                point_land_days,
                point_missing_bathy,
                point_skipped_weather,
            )

        self._output.optimize()

        logger.success(
            "Wave activity database ready: "
            "points={}, processed_days={}, active_rows={}, "
            "skipped_weather={}, land_sector_days={}, "
            "missing_bathymetry_days={}",
            total_points,
            total_processed_days,
            total_activity,
            skipped_weather,
            land_sector_rows,
            missing_bathymetry,
        )

        return WaveActivityBuildStats(
            point_count=total_points,
            processed_day_count=total_processed_days,
            activity_count=total_activity,
            skipped_weather_rows=skipped_weather,
            land_sector_rows=land_sector_rows,
            missing_bathymetry_rows=missing_bathymetry,
        )

    @staticmethod
    def _prepare_weather_record(
        wind_speed_max,
        wind_direction,
    ) -> tuple[float, int] | None:
        if (
            wind_speed_max is None
            or wind_direction is None
        ):
            return None

        try:
            speed = float(wind_speed_max)
            direction_value = float(wind_direction)
        except (TypeError, ValueError):
            return None

        if (
            not math.isfinite(speed)
            or not math.isfinite(direction_value)
        ):
            return None

        direction = (
            int(round(direction_value)) % 360
        )

        return max(speed, 0.0), direction

    def _point_source_id(self) -> int:
        value = self._session.execute(
            select(
                CoastlineNormalSourceModel
                .point_source_id
            ).where(
                CoastlineNormalSourceModel.id
                == self._normal_source_id
            )
        ).scalar_one()

        return int(value)

    def _fetch_lookup(
        self,
        point_id: int,
    ) -> FetchLookup:
        rows = self._session.execute(
            select(WindFetchModel)
            .where(
                WindFetchModel.point_id
                == point_id
            )
            .order_by(
                WindFetchModel.azimuth_deg
            )
        ).scalars().all()

        if not rows:
            raise ValueError(
                "No wind fetches found for "
                f"point_id={point_id}"
            )

        directions: list[float] = []
        fetches: list[float] = []

        for row in rows:
            direction = (
                float(row.azimuth_deg) % 360.0
            )
            fetch_m = float(
                row.fetch_length_m
            )

            if (
                not math.isfinite(fetch_m)
                or fetch_m <= 0.0
            ):
                raise ValueError(
                    "Invalid fetch length: "
                    f"point_id={point_id}, "
                    f"direction={direction}, "
                    f"fetch_m={fetch_m}"
                )

            directions.append(direction)
            fetches.append(fetch_m)

        return FetchLookup(
            pd.DataFrame(
                {
                    "direction": directions,
                    "fetch_m": fetches,
                }
            )
        )
