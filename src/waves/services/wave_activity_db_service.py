from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
from src.waves.nearshore import NearshoreWaveTransformer
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
    activity_count: int
    skipped_weather_rows: int
    skipped_no_fetch_rows: int
    fallback_bathy_rows: int


def load_wave_points(
    session: Session,
    normal_source_id: int,
) -> list[CoastlineWavePoint]:
    normal_source = session.get(CoastlineNormalSourceModel, normal_source_id)
    if normal_source is None:
        raise ValueError(f"CoastlineNormalSource id={normal_source_id} not found")

    rows = session.execute(
        select(CoastlineNormalModel, CoastlinePointModel)
        .join(
            CoastlinePointModel,
            CoastlineNormalModel.point_id == CoastlinePointModel.id,
        )
        .where(CoastlineNormalModel.normal_source_id == normal_source_id)
        .order_by(CoastlinePointModel.seq, CoastlinePointModel.id)
    ).all()

    return [
        CoastlineWavePoint(
            point_id=int(point.id),
            lon=float(point.lon),
            lat=float(point.lat),
            normal_x=float(normal.nx),
            normal_y=float(normal.ny),
            normal_azimuth_deg=float(normal.normal_azimuth_deg) % 360.0,
        )
        for normal, point in rows
    ]


def resolve_normal_source_id(
    session: Session,
    normal_source_id: int | None,
) -> int:
    if normal_source_id is not None:
        if session.get(CoastlineNormalSourceModel, normal_source_id) is None:
            raise ValueError(
                f"CoastlineNormalSource id={normal_source_id} not found"
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
            WindFetchModel.point_id == CoastlineNormalModel.point_id,
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


def _load_fetch_directions(session: Session, point_id: int) -> set[int]:
    """Возвращает множество азимутов (целых градусов), для которых есть фетч."""
    rows = session.execute(
        select(WindFetchModel.azimuth_deg)
        .where(WindFetchModel.point_id == point_id)
    ).scalars().all()
    return {int(round(az)) % 360 for az in rows}


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
        bathy_n_steps: int = 200,
        commit_batch_size: int = 5_000,
        default_h_deep_m: float = 20.0,
        default_h_point_m: float = 3.0,
    ) -> None:
        if overwater_factor <= 0:
            raise ValueError("overwater_factor must be > 0")
        if bathy_n_steps < 2:
            raise ValueError("bathy_n_steps must be >= 2")
        if commit_batch_size < 1:
            raise ValueError("commit_batch_size must be >= 1")
        if default_h_deep_m <= 0:
            raise ValueError("default_h_deep_m must be > 0")
        if default_h_point_m <= 0:
            raise ValueError("default_h_point_m must be > 0")

        self._session = coastline_session
        self._weather = weather_repository
        self._bathymetry = bathymetry_archive
        self._output = output_repository
        self._normal_source_id = normal_source_id
        self._start_date = start_date
        self._end_date = end_date
        self._overwater_factor = overwater_factor
        self._bathy_n_steps = bathy_n_steps
        self._commit_batch_size = commit_batch_size
        self._default_h_deep_m = default_h_deep_m
        self._default_h_point_m = default_h_point_m
        self._smb = SMBWaveGrowthModel()
        self._energy = WaveEnergyCalculator()

    def run(self) -> WaveActivityBuildStats:
        points = load_wave_points(self._session, self._normal_source_id)
        if not points:
            raise ValueError(
                f"Normal source id={self._normal_source_id} contains no points"
            )

        if self._bathymetry.point_source_id != self._point_source_id():
            raise ValueError(
                "Bathymetry archive and selected normal source refer to "
                "different coastline point sources"
            )

        self._output.initialize()
        total_activity = 0
        skipped_weather = 0
        skipped_no_fetch = 0
        fallback_bathy = 0
        total_points = len(points)

        logger.info(
            "Wave activity calculation started: points={}, normal_source_id={}",
            total_points,
            self._normal_source_id,
        )

        for index, point in enumerate(points, start=1):
            logger.info(
                "[{}/{}] point_id={}: loading fetches and weather",
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
                    normal_azimuth_deg=point.normal_azimuth_deg,
                )
            )

            fetch_directions = _load_fetch_directions(self._session, point.point_id)
            if not fetch_directions:
                logger.warning(
                    "[{}/{}] point_id={}: no fetch directions found, skipping",
                    index,
                    total_points,
                    point.point_id,
                )
                self._output.add_summary(
                    WaveActivitySummaryRow.from_powers(point.point_id, [])
                )
                self._output.commit()
                continue

            fetch_lookup = self._fetch_lookup(point.point_id)
            bathy_service = self._bathymetry.for_point_id(
                point.point_id,
                n_steps=self._bathy_n_steps,
            )

            # Трансформер с батиметрией — основной
            transformer = NearshoreWaveTransformer(
                shore_normal_deg=point.normal_azimuth_deg,
                profile_provider=bathy_service,
                default_h_deep_m=self._default_h_deep_m,
                default_h_point_m=self._default_h_point_m,
            )
            # Трансформер без батиметрии — fallback для направлений
            # с пустым/нулевым профилем
            fallback_transformer = NearshoreWaveTransformer(
                shore_normal_deg=point.normal_azimuth_deg,
                profile_provider=None,
                default_h_deep_m=self._default_h_deep_m,
                default_h_point_m=self._default_h_point_m,
            )

            weather = self._weather.get_interpolated_timeseries(
                point.lat,
                point.lon,
                self._start_date,
                self._end_date,
            )

            pending: list[WaveActivityRow] = []
            powers: list[int] = []

            for record in weather:
                if record.wind_speed_max is None or record.wind_direction is None:
                    skipped_weather += 1
                    continue

                direction = int(round(record.wind_direction)) % 360

                if direction not in fetch_directions:
                    skipped_no_fetch += 1
                    continue

                wind_kmh = max(float(record.wind_speed_max), 0.0)
                if wind_kmh < 0.1:
                    power_i = 0
                else:
                    wind_ms = wind_kmh / 3.6 * self._overwater_factor
                    fetch_m = fetch_lookup.get_fetch(direction)
                    hs_offshore, tp_s = self._smb.calculate(wind_ms, fetch_m)

                    try:
                        nearshore = transformer.transform(
                            direction, hs_offshore, tp_s
                        )
                    except WaveBathymetryError:
                        # Профиль по этому направлению не содержит валидных
                        # глубин — используем дефолтные глубины разрушения волны
                        fallback_bathy += 1
                        logger.debug(
                            "point_id={}, direction={}°: "
                            "bathymetry fallback (h_deep={}, h_point={})",
                            point.point_id,
                            direction,
                            self._default_h_deep_m,
                            self._default_h_point_m,
                        )
                        nearshore = fallback_transformer.transform(
                            direction, hs_offshore, tp_s
                        )

                    power = self._energy.cwef(
                        self._energy.wave_power(nearshore.hs_nearshore_m, tp_s),
                        nearshore.cos_shore,
                    )
                    power_i = max(0, round(power))

                # powers собирает все значения для корректного расчёта summary
                powers.append(power_i)

                # В БД записываем только дни с ненулевой мощностью
                if power_i > 0:
                    pending.append(
                        WaveActivityRow(
                            point_id=point.point_id,
                            day=encode_day(record.obs_date),
                            wind_azimuth_deg=direction,
                            wave_power_wm=power_i,
                        )
                    )

                if len(pending) >= self._commit_batch_size:
                    total_activity += self._output.add_activity(pending)
                    pending.clear()
                    self._output.commit()

            total_activity += self._output.add_activity(pending)
            self._output.add_summary(
                WaveActivitySummaryRow.from_powers(point.point_id, powers)
            )
            self._output.commit()

            logger.info(
                "[{}/{}] point_id={}: days={}, active_days={}, "
                "skipped_no_fetch={}, fallback_bathy={}",
                index,
                total_points,
                point.point_id,
                len(powers),
                sum(v > 0 for v in powers),
                skipped_no_fetch,
                fallback_bathy,
            )

        self._output.optimize()
        logger.success(
            "Wave activity database ready: points={}, rows={}, "
            "skipped_weather={}, skipped_no_fetch={}, fallback_bathy={}",
            total_points,
            total_activity,
            skipped_weather,
            skipped_no_fetch,
            fallback_bathy,
        )
        return WaveActivityBuildStats(
            point_count=total_points,
            activity_count=total_activity,
            skipped_weather_rows=skipped_weather,
            skipped_no_fetch_rows=skipped_no_fetch,
            fallback_bathy_rows=fallback_bathy,
        )

    def _point_source_id(self) -> int:
        value = self._session.execute(
            select(CoastlineNormalSourceModel.point_source_id).where(
                CoastlineNormalSourceModel.id == self._normal_source_id
            )
        ).scalar_one()
        return int(value)

    def _fetch_lookup(self, point_id: int) -> FetchLookup:
        import pandas as pd

        rows = self._session.execute(
            select(WindFetchModel)
            .where(WindFetchModel.point_id == point_id)
            .order_by(WindFetchModel.azimuth_deg)
        ).scalars().all()
        if not rows:
            raise ValueError(f"No wind fetches found for point_id={point_id}")

        return FetchLookup(
            pd.DataFrame(
                {
                    "direction": [row.azimuth_deg for row in rows],
                    "fetch_m": [row.fetch_length_m for row in rows],
                }
            )
        )
