from collections import OrderedDict
from dataclasses import dataclass
import math
import time

import numpy as np

from loguru import logger
from pyproj import Transformer
from rasterio.transform import rowcol

from shapely.geometry import LineString, Point

from src.weather_history.archive.repository import (
    CompactWeatherRepository,
)

from src.waves.energy import WaveEnergyCalculator
from src.waves.errors import WaveBathymetryError

from src.waves.nearshore import (
    BreakingModel,
    NearshoreWaveTransformer,
    RefractionModel,
)

from src.waves.offshore import SMBWaveGrowthModel

from .bathymetry_policy import POLICY
from .geometry import geometry_parts


# ============================================================
# LRU-КЕШ
# ============================================================


class LRU:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError(
                "Размер LRU-кеша должен быть целым > 0"
            )

        self.capacity = capacity
        self.items = OrderedDict()

        self.hits = 0
        self.misses = 0

    def contains(self, key):
        return key in self.items

    def get(self, key, factory):
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]

        self.misses += 1

        # Незавершённые вычисления и исключения
        # не попадают в кеш.
        value = factory()

        self.items[key] = value

        while len(self.items) > self.capacity:
            self.items.popitem(last=False)

        return value


# ============================================================
# МОДЕЛИ ДАННЫХ
# ============================================================


@dataclass
class Ray:
    line: object
    projected_distances: np.ndarray
    geodesic_distances: np.ndarray
    original_fetch_m: float


@dataclass(frozen=True)
class RasterProfile:
    direction: int

    distances_m: np.ndarray
    depths_m: np.ndarray

    finite_fraction: float
    wet_fraction: float

    outside_count: int
    pixel_count: int
    wet_pixel_count: int

    reason: str

    @property
    def has_depths(self):
        return bool(
            (
                np.isfinite(self.depths_m)
                & (self.depths_m > 0)
            ).any()
        )


class PointProfileProvider:
    """Адаптер к NearshoreWaveTransformer исходного проекта."""

    def __init__(
        self,
        physics,
        site,
        scenario,
        fetch_table,
    ):
        self.physics = physics
        self.site = site
        self.scenario = scenario
        self.fetch_table = fetch_table

    def get_profile(self, direction):
        index = self.physics.direction_index(
            direction
        )

        return self.physics.profile(
            self.site,
            int(self.physics.azimuths[index]),
            float(self.fetch_table[index]),
            scenario=self.scenario,
        )


# ============================================================
# ФИЗИЧЕСКИЙ РАСЧЁТ
# ============================================================


class Physics:
    def __init__(self, cfg, geometry):
        self.cfg = cfg
        self.geometry = geometry
        self.policy = POLICY

        self.weather = None
        self.raster = None
        self.to_raster = None

        self.weather_cache = LRU(
            cfg.weather_cache_size
        )
        self.ray_cache = LRU(
            cfg.ray_cache_size
        )
        self.profile_cache = LRU(
            cfg.profile_cache_size
        )

        self.azimuths = (
            np.arange(
                cfg.direction_count,
                dtype=int,
            )
            * (360 // cfg.direction_count)
        )

        self.smb = SMBWaveGrowthModel(
            g=cfg.gravity
        )

        self.energy = WaveEnergyCalculator(
            rho_water=cfg.rho_water,
            g=cfg.gravity,
        )

        try:
            self.policy.validate()

            self.weather = CompactWeatherRepository(
                cfg.weather_path
            )

            logger.info(
                "Погодный архив открыт: {}",
                cfg.weather_path,
            )

            if cfg.bathymetry_path is not None:
                import rasterio

                self.raster = rasterio.open(
                    cfg.bathymetry_path
                )

                if self.raster.crs is None:
                    raise ValueError(
                        "У батиметрии не задана CRS"
                    )

                self.to_raster = Transformer.from_crs(
                    4326,
                    self.raster.crs,
                    always_xy=True,
                )

                logger.info(
                    "Батиметрия открыта: {} | "
                    "{}x{} | CRS={} | шаг={}",
                    cfg.bathymetry_path,
                    self.raster.width,
                    self.raster.height,
                    self.raster.crs,
                    self.raster.res,
                )

            elif cfg.bathymetry_required:
                raise ValueError(
                    "Батиметрический растр обязателен, "
                    "но не подготовлен"
                )

            else:
                logger.warning(
                    "Реальный батиметрический растр "
                    "не используется"
                )

            logger.info(
                "Политика отсутствующей батиметрии: {}",
                self.policy.mode,
            )

            logger.info(
                "Измеренный профиль принимается при "
                "наличии хотя бы одной положительной глубины; "
                "порог доли морских отсчётов не используется"
            )

            if self.policy.mode == "default_depths":
                logger.warning(
                    "При отсутствии глубин используется "
                    "ЯВНАЯ МОДЕЛЬ постоянных глубин: "
                    "reference={:.2f} м, coastal={:.2f} м. "
                    "Затронутые дни отмечаются в GeoJSON.",
                    self.policy.reference_depth_m,
                    self.policy.coastal_depth_m,
                )

            elif self.policy.mode == "repository_zero":
                logger.warning(
                    "Режим исходного репозитория: "
                    "дни без батиметрии получают CWEF=0. "
                    "В минимизирующем ГА это может "
                    "искусственно улучшать кандидатов."
                )

        except BaseException:
            self.close()
            raise

    def close(self):
        if self.weather is not None:
            self.weather.close()
            self.weather = None

        if self.raster is not None:
            self.raster.close()
            self.raster = None

    def direction_index(self, direction):
        step = (
            360 // self.cfg.direction_count
        )

        return int(
            math.floor(
                (
                    float(direction) % 360
                    + step / 2
                ) / step
            )
            % self.cfg.direction_count
        )

    # ========================================================
    # ПОГОДА
    # ========================================================

    def weather_series(self, site):
        def load():
            started = time.perf_counter()

            lon, lat = (
                self.geometry.to_lonlat.transform(
                    site.x,
                    site.y,
                )
            )

            records = (
                self.weather.get_interpolated_timeseries(
                    lat,
                    lon,
                    self.cfg.start_date,
                    self.cfg.end_date,
                )
            )

            speeds = []
            directions = []
            dates = []
            skipped = 0

            for record in records:
                if (
                    record.wind_speed_max is None
                    or record.wind_direction is None
                ):
                    skipped += 1
                    continue

                speed = float(
                    record.wind_speed_max
                )
                direction = float(
                    record.wind_direction
                )

                if (
                    not math.isfinite(speed)
                    or not math.isfinite(direction)
                    or speed < 0
                ):
                    skipped += 1
                    continue

                speeds.append(speed)

                directions.append(
                    int(round(direction)) % 360
                )

                dates.append(
                    record.obs_date.isoformat()
                )

            if not speeds:
                raise ValueError(
                    "Нет валидных погодных дней: "
                    f"lon={lon}, lat={lat}"
                )

            logger.debug(
                "Погода lon={:.7f}, lat={:.7f} | "
                "дней={} | пропущено={} | {:.3f} с",
                lon,
                lat,
                len(speeds),
                skipped,
                time.perf_counter() - started,
            )

            return (
                np.asarray(
                    speeds,
                    dtype=float,
                ),
                np.asarray(
                    directions,
                    dtype=int,
                ),
                dates[0],
                dates[-1],
                skipped,
            )

        return self.weather_cache.get(
            (site.x, site.y),
            load,
        )

    # ========================================================
    # РАЗГОНЫ
    # ========================================================

    def first_hit_distance(self, ray, obstacle):
        intersection = (
            ray.line.intersection(obstacle)
        )

        if intersection.is_empty:
            return self.cfg.max_fetch_m

        positions = []

        for component in geometry_parts(
            intersection,
            {"Point", "LineString"},
        ):
            coordinates = (
                [component.coords[0]]
                if component.geom_type == "Point"
                else list(component.coords)
            )

            positions.extend(
                ray.line.project(
                    Point(coordinate)
                )
                for coordinate in coordinates
            )

        if not positions:
            raise ValueError(
                "Не удалось определить "
                "пересечение fetch-луча"
            )

        return float(
            np.interp(
                min(positions),
                ray.projected_distances,
                ray.geodesic_distances,
            )
        )

    def original_rays(self, site):
        def build():
            started = time.perf_counter()

            lon, lat = (
                self.geometry.to_lonlat.transform(
                    site.x,
                    site.y,
                )
            )

            distances = np.linspace(
                self.cfg.probe_m,
                self.cfg.max_fetch_m,
                math.ceil(
                    self.cfg.max_fetch_m
                    / self.cfg.geodesic_step_m
                ) + 1,
            )

            rays = []

            for azimuth in self.azimuths:
                lons, lats, _ = (
                    self.geometry.geod.fwd(
                        np.full(
                            len(distances),
                            lon,
                        ),
                        np.full(
                            len(distances),
                            lat,
                        ),
                        np.full(
                            len(distances),
                            float(azimuth),
                        ),
                        distances,
                    )
                )

                xs, ys = (
                    self.geometry.to_metric.transform(
                        lons,
                        lats,
                    )
                )

                xs = np.asarray(
                    xs,
                    dtype=float,
                )
                ys = np.asarray(
                    ys,
                    dtype=float,
                )

                if not (
                    np.isfinite(xs).all()
                    and np.isfinite(ys).all()
                ):
                    raise ValueError(
                        "Невалидные координаты fetch-луча"
                    )

                ray = Ray(
                    line=LineString(
                        np.column_stack(
                            [xs, ys]
                        )
                    ),
                    projected_distances=np.r_[
                        0.0,
                        np.cumsum(
                            np.hypot(
                                np.diff(xs),
                                np.diff(ys),
                            )
                        ),
                    ],
                    geodesic_distances=distances,
                    original_fetch_m=0.0,
                )

                if self.geometry.original_water(
                    Point(
                        xs[0],
                        ys[0],
                    )
                ):
                    ray.original_fetch_m = (
                        self.first_hit_distance(
                            ray,
                            self.geometry.fetch_obstacles,
                        )
                    )

                rays.append(ray)

            logger.debug(
                "Исходные fetch-лучи: {} | {:.3f} с",
                len(rays),
                time.perf_counter() - started,
            )

            return rays

        return self.ray_cache.get(
            (site.x, site.y),
            build,
        )

    def scenario_fetches(self, site, scenario):
        values = []

        for ray in self.original_rays(site):
            start = Point(
                ray.line.coords[0]
            )

            if not self.geometry.scenario_water(
                start,
                scenario.fill,
            ):
                values.append(0.0)
                continue

            fetch = ray.original_fetch_m

            if not scenario.fill.is_empty:
                fetch = min(
                    fetch,
                    self.first_hit_distance(
                        ray,
                        scenario.fill.boundary,
                    ),
                )

            values.append(fetch)

        return np.asarray(
            values,
            dtype=float,
        )

    # ========================================================
    # БАТИМЕТРИЧЕСКИЙ ПРОФИЛЬ
    # ========================================================

    def profile(self, site, azimuth, fetch, scenario=None):
        """Профиль до текущего fetch.

        Неположительные высоты преобразуются:
        elevation < 0 -> положительная глубина;
        elevation >= 0 -> NaN.

        Отсутствие морских отсчётов возвращается
        как пустой по глубинам профиль, а не как ValueError.
        """
        def sample():
            if not math.isfinite(fetch) or fetch <= 0:
                return RasterProfile(
                    direction=int(azimuth),
                    distances_m=np.empty(0),
                    depths_m=np.empty(0),
                    finite_fraction=0.0,
                    wet_fraction=0.0,
                    outside_count=0,
                    pixel_count=0,
                    wet_pixel_count=0,
                    reason="nonpositive_fetch",
                )

            if self.raster is None:
                return RasterProfile(
                    direction=int(azimuth),
                    distances_m=np.empty(0),
                    depths_m=np.empty(0),
                    finite_fraction=0.0,
                    wet_fraction=0.0,
                    outside_count=0,
                    pixel_count=0,
                    wet_pixel_count=0,
                    reason="raster_not_configured",
                )

            lon, lat = (
                self.geometry.to_lonlat.transform(
                    site.x,
                    site.y,
                )
            )

            begin = min(
                self.cfg.probe_m,
                fetch * 0.1,
            )

            end = max(
                begin,
                fetch - min(
                    self.cfg.probe_m,
                    fetch * 0.01,
                ),
            )

            distances = np.linspace(
                begin,
                end,
                self.cfg.bathymetry_samples,
            )

            lons, lats, _ = (
                self.geometry.geod.fwd(
                    np.full(
                        len(distances),
                        lon,
                    ),
                    np.full(
                        len(distances),
                        lat,
                    ),
                    np.full(
                        len(distances),
                        float(azimuth),
                    ),
                    distances,
                )
            )

            xs, ys = self.to_raster.transform(
                lons,
                lats,
            )

            xs = np.asarray(
                xs,
                dtype=float,
            ).reshape(-1)

            ys = np.asarray(
                ys,
                dtype=float,
            ).reshape(-1)

            if not (
                np.isfinite(xs).all()
                and np.isfinite(ys).all()
            ):
                raise ValueError(
                    "Невалидные координаты "
                    "батиметрического профиля"
                )

            rows, columns = rowcol(
                self.raster.transform,
                xs,
                ys,
                op=np.floor,
            )

            rows = np.asarray(
                rows,
                dtype=np.int64,
            ).reshape(-1)

            columns = np.asarray(
                columns,
                dtype=np.int64,
            ).reshape(-1)

            inside = (
                (rows >= 0)
                & (rows < self.raster.height)
                & (columns >= 0)
                & (columns < self.raster.width)
            )

            # Выход за покрытие — проблема архива,
            # а не локального несовпадения берега с пикселем.
            if not inside.all():
                raise ValueError(
                    "Профиль выходит за покрытие растра: "
                    f"lon={lon:.7f}, lat={lat:.7f}, "
                    f"azimuth={azimuth}, fetch={fetch:.1f} м. "
                    "Расширьте покрытие батиметрии."
                )

            elevation = (
                np.ma.asarray(
                    list(
                        self.raster.sample(
                            zip(xs, ys),
                            indexes=1,
                            masked=True,
                        )
                    )
                )
                .astype(float)
                .filled(np.nan)
                .reshape(-1)
            )

            finite = np.isfinite(elevation)
            wet = finite & (elevation < 0)

            depths = np.full(
                elevation.shape,
                np.nan,
                dtype=float,
            )
            depths[wet] = -elevation[wet]

            pixel_ids = np.column_stack(
                [rows, columns]
            )

            pixel_count = len(
                np.unique(
                    pixel_ids,
                    axis=0,
                )
            )

            wet_pixel_count = (
                len(
                    np.unique(
                        pixel_ids[wet],
                        axis=0,
                    )
                )
                if wet.any()
                else 0
            )

            if wet.any():
                reason = "measured"
            elif finite.any():
                reason = "no_negative_elevation"
            else:
                reason = "nodata"

            result = RasterProfile(
                direction=int(azimuth),
                distances_m=distances,
                depths_m=depths,
                finite_fraction=float(
                    finite.mean()
                ),
                wet_fraction=float(
                    wet.mean()
                ),
                outside_count=0,
                pixel_count=int(
                    pixel_count
                ),
                wet_pixel_count=int(
                    wet_pixel_count
                ),
                reason=reason,
            )

            logger.debug(
                "Профиль az={}° | fetch={:.2f} м | "
                "finite={:.1%} | wet={:.1%} | "
                "пикселей={} | морских={} | status={}",
                azimuth,
                fetch,
                result.finite_fraction,
                result.wet_fraction,
                result.pixel_count,
                result.wet_pixel_count,
                result.reason,
            )

            return result

        # Профиль зависит от координат, направления и fetch.
        # Поиск соседних пикселей и геометрические обходы
        # больше не используются.
        return self.profile_cache.get(
            (
                site.x,
                site.y,
                int(azimuth),
                float(fetch),
            ),
            sample,
        )

    # ========================================================
    # CWEF ОДНОЙ ТОЧКИ
    # ========================================================

    def point_metrics(self, site, scenario):
        started = time.perf_counter()

        (
            speeds,
            directions,
            first_date,
            last_date,
            skipped,
        ) = self.weather_series(site)

        fetch_table = self.scenario_fetches(
            site,
            scenario,
        )

        bins = np.asarray(
            [
                self.direction_index(direction)
                for direction in directions
            ],
            dtype=int,
        )

        fetches = fetch_table[bins]

        delta = (
            directions - site.normal_deg + 180
        ) % 360 - 180

        sea = np.abs(delta) <= 90
        windy = speeds >= 0.1

        active = (
            sea
            & windy
            & (fetches > 0)
        )

        cwef = np.zeros(
            len(speeds),
            dtype=float,
        )

        profile_provider = PointProfileProvider(
            physics=self,
            site=site,
            scenario=scenario,
            fetch_table=fetch_table,
        )

        transformer = NearshoreWaveTransformer(
            shore_normal_deg=site.normal_deg,
            profile_provider=profile_provider,
            breaking_model=BreakingModel(
                gamma_b=self.cfg.breaking_coefficient
            ),
            refraction_model=RefractionModel(
                g=self.cfg.gravity
            ),
        )

        fallback_transformer = NearshoreWaveTransformer(
            shore_normal_deg=site.normal_deg,
            profile_provider=None,
            default_h_deep_m=(
                self.policy.reference_depth_m
            ),
            default_h_point_m=(
                self.policy.coastal_depth_m
            ),
            breaking_model=BreakingModel(
                gamma_b=self.cfg.breaking_coefficient
            ),
            refraction_model=RefractionModel(
                g=self.cfg.gravity
            ),
        )

        missing_direction_mask = np.zeros(
            len(speeds),
            dtype=bool,
        )

        profiles = {}

        for index in np.unique(
            bins[active]
        ):
            profile = self.profile(
                site,
                int(self.azimuths[index]),
                float(fetch_table[index]),
                scenario=scenario,
            )

            profiles[int(index)] = profile

            if not profile.has_depths:
                affected = (
                    active & (bins == index)
                )

                missing_direction_mask |= affected

                logger.warning(
                    "НЕТ БАТИМЕТРИИ | "
                    "x={:.2f}, y={:.2f}, az={}° | "
                    "fetch={:.2f} м | причина={} | "
                    "активных дней={} | политика={}",
                    site.x,
                    site.y,
                    int(self.azimuths[index]),
                    float(fetch_table[index]),
                    profile.reason,
                    int(affected.sum()),
                    self.policy.mode,
                )

                if self.policy.mode == "strict":
                    raise WaveBathymetryError(
                        "Отсутствует батиметрия: "
                        f"x={site.x}, y={site.y}, "
                        f"azimuth={self.azimuths[index]}, "
                        f"fetch={fetch_table[index]}, "
                        f"reason={profile.reason}"
                    )

        active_indices = np.flatnonzero(active)

        # Повторяющиеся сочетания скорости и направления
        # имеют одинаковые fetch и профиль в одной точке.
        if len(active_indices):
            combinations = np.column_stack(
                [
                    speeds[active],
                    directions[active],
                ]
            )

            unique, inverse = np.unique(
                combinations,
                axis=0,
                return_inverse=True,
            )

            powers = np.zeros(
                len(unique),
                dtype=float,
            )

            for group_index, combination in enumerate(unique):
                speed = float(combination[0])
                direction = int(combination[1])

                fetch_index = self.direction_index(
                    direction
                )

                fetch = float(
                    fetch_table[fetch_index]
                )

                profile = profiles[
                    fetch_index
                ]

                missing = not profile.has_depths

                if (
                    missing
                    and self.policy.mode == "repository_zero"
                ):
                    powers[group_index] = 0.0
                    continue

                hs_offshore, period = self.smb.calculate(
                    speed
                    / 3.6
                    * self.cfg.overwater_factor,
                    fetch,
                )

                if missing:
                    nearshore = fallback_transformer.transform(
                        direction,
                        hs_offshore,
                        period,
                    )
                else:
                    # При наличии хотя бы одной глубины
                    # исходный transformer сам отбирает
                    # положительные конечные значения.
                    nearshore = transformer.transform(
                        direction,
                        hs_offshore,
                        period,
                    )

                power = self.energy.cwef(
                    self.energy.wave_power(
                        nearshore.hs_nearshore_m,
                        period,
                    ),
                    nearshore.cos_shore,
                )

                if not math.isfinite(power) or power < 0:
                    raise ValueError(
                        "Физическая модель вернула "
                        "невалидный CWEF: "
                        f"{power}"
                    )

                powers[group_index] = power

            cwef[active_indices] = powers[inverse]

        missing_days = int(
            missing_direction_mask.sum()
        )

        fallback_days = (
            missing_days
            if self.policy.mode == "default_depths"
            else 0
        )

        zero_missing_days = (
            missing_days
            if self.policy.mode == "repository_zero"
            else 0
        )

        measured_profiles = [
            profile
            for profile in profiles.values()
            if profile.has_depths
        ]

        finite_fractions = [
            profile.finite_fraction
            for profile in profiles.values()
        ]

        wet_fractions = [
            profile.wet_fraction
            for profile in profiles.values()
        ]

        result = {
            # Все валидные погодные дни входят в среднее,
            # включая штиль и сухопутные направления.
            "mean_CWEF_Wm": float(
                cwef.mean()
            ),
            "p95_CWEF_Wm": float(
                np.percentile(cwef, 95)
            ),
            "max_CWEF_Wm": float(
                cwef.max()
            ),
            "std_CWEF_Wm": (
                float(cwef.std(ddof=1))
                if len(cwef) > 1
                else 0.0
            ),

            "n_valid_days": int(
                len(cwef)
            ),
            "n_active_days": int(
                (cwef > 0).sum()
            ),
            "n_skipped_days": int(
                skipped
            ),

            "n_land_sector_days": int(
                (~sea).sum()
            ),
            "n_zero_fetch_sea_days": int(
                (sea & windy & (fetches <= 0)).sum()
            ),

            "n_bathymetry_directions": int(
                len(profiles)
            ),
            "n_measured_bathymetry_directions": int(
                len(measured_profiles)
            ),
            "n_missing_bathymetry_directions": int(
                len(profiles) - len(measured_profiles)
            ),

            "n_missing_bathymetry_days": missing_days,
            "n_default_depth_days": fallback_days,
            "n_repository_zero_bathymetry_days": (
                zero_missing_days
            ),

            "default_depth_days_fraction": float(
                fallback_days / len(cwef)
            ),
            "missing_bathymetry_active_days_fraction": (
                float(missing_days / int(active.sum()))
                if active.any()
                else 0.0
            ),

            "n_single_wet_pixel_profiles": int(
                sum(
                    profile.wet_pixel_count == 1
                    for profile in measured_profiles
                )
            ),

            "min_profile_finite_fraction": (
                float(min(finite_fractions))
                if finite_fractions
                else None
            ),
            "min_profile_wet_fraction": (
                float(min(wet_fractions))
                if wet_fractions
                else None
            ),

            "bathymetry_policy": self.policy.mode,
            "bathymetry_used": self.raster is not None,
            "default_depths_used": bool(
                fallback_days
            ),

            "fallback_reference_depth_m": (
                self.policy.reference_depth_m
                if fallback_days
                else None
            ),
            "fallback_coastal_depth_m": (
                self.policy.coastal_depth_m
                if fallback_days
                else None
            ),

            "first_date": first_date,
            "last_date": last_date,
        }

        if missing_days:
            logger.info(
                "КАЧЕСТВО ТОЧКИ | x={:.2f}, y={:.2f} | "
                "направлений без глубин={}/{} | "
                "дней без глубин={} | fallback={} | "
                "repository_zero={}",
                site.x,
                site.y,
                result["n_missing_bathymetry_directions"],
                result["n_bathymetry_directions"],
                missing_days,
                fallback_days,
                zero_missing_days,
            )

        logger.debug(
            "CWEF точки | x={:.2f}, y={:.2f}, normal={:.2f}° | "
            "дней={} | mean={:.6f} Вт/м | {:.3f} с",
            site.x,
            site.y,
            site.normal_deg,
            len(cwef),
            result["mean_CWEF_Wm"],
            time.perf_counter() - started,
        )

        return result