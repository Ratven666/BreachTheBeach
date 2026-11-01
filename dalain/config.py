from dataclasses import dataclass
from datetime import date
from pathlib import Path
import math


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    # ========================================================
    # ВХОДНЫЕ И ВЫХОДНЫЕ ДАННЫЕ
    # ========================================================

    coastline_path: Path = (
        PROJECT_ROOT
        / "Kaliningrad/data/coastline/"
          "kaliningrad_OFFSET_merged.geojson"
    )

    weather_path: Path = (
        PROJECT_ROOT
        / "Kaliningrad/data/db/weather_compact.db"
    )

    output_directory: Path = PROJECT_ROOT / "dalain/output"

    # Polygon/MultiPolygon исходной суши.
    # None — классификация по нормалям ближайшего берега.
    land_path: Path | None = None

    # Готовый GeoTIFF elevation: дно отрицательное.
    # None — автоматическая загрузка, если она включена.
    bathymetry_path: Path | None = None

    # ========================================================
    # ГЕОМЕТРИЯ
    # ========================================================

    # Замените своими координатами.
    center_lon: float = 19.9
    center_lat: float = 54.7

    target_area_m2: float = 100_000.0
    cell_side_m: float = 25.0

    search_radius_m: float = 2_000.0
    evaluation_radius_m: float = 2_500.0

    connection: str = "attached"  # attached | island
    sea_side: str = "right"       # left | right

    point_spacing_m: float = 50.0
    tangent_delta_m: float = 30.0

    probe_m: float = 0.10
    tolerance_m: float = 0.001

    max_grid_cells: int = 2_000_000

    # ========================================================
    # ПОГОДА И ВОЛНЫ
    # ========================================================

    # Обе границы включительно. None — без ограничения.
    start_date: str | None = None
    end_date: str | None = None

    # 360 — через 1 градус; 72 — через 5 градусов.
    direction_count: int = 360

    max_fetch_m: float = 100_000.0
    geodesic_step_m: float = 1_000.0

    overwater_factor: float = 1.1

    rho_water: float = 1025.0
    gravity: float = 9.81

    # Только для явно разрешённого режима без батиметрии.
    reference_depth_m: float = 20.0
    coastal_depth_m: float = 3.0

    breaking_coefficient: float = 0.55

    # ========================================================
    # БАТИМЕТРИЧЕСКИЕ ПРОФИЛИ
    # ========================================================

    bathymetry_samples: int = 300

    # Наличие данных: конечные значения, включая сушу.
    min_finite_bathymetry_fraction: float = 0.95

    # Минимальная доля собственных морских отсчётов.
    min_valid_bathymetry_fraction: float = 0.25

    # Допуск коротких профилей с малой долей морских отсчётов.
    allow_underresolved_bathymetry: bool = True
    short_profile_pixel_diagonals: float = 2.0

    # Если короткий профиль вообще не содержит морских
    # отсчётов, разрешить поиск реального морского пикселя.
    allow_nearby_wet_pixel: bool = True

    # Радиус от середины профиля в диагоналях пикселя.
    nearby_wet_pixel_radius_diagonals: float = 1.0

    # Полуразмер окна поиска в пикселях.
    # Радиус в метрах проверяется дополнительно.
    nearby_wet_pixel_window_radius: int = 3

    # ========================================================
    # СКАЧИВАНИЕ БАТИМЕТРИИ
    # ========================================================

    bathymetry_auto_download: bool = True
    bathymetry_required: bool = True

    bathymetry_provider: str = "auto"  # auto | emodnet | gebco

    bathymetry_cache_directory: Path = (
        PROJECT_ROOT / "dalain/data/bathymetry"
    )

    bathymetry_force_download: bool = False

    bathymetry_download_retries: int = 3
    bathymetry_download_timeout_s: int = 300

    bathymetry_safety_factor: float = 1.05
    bathymetry_padding_m: float = 2_000.0

    bathymetry_max_cells_per_tile: int = 2_000_000
    bathymetry_planning_resolution_arcsec: float = 3.75
    bathymetry_tile_overlap_cells: int = 3
    bathymetry_merge_memory_mb: int = 128

    # 1.0 — elevation уже отрицательная на дне.
    # -1.0 — источник возвращает положительные глубины.
    bathymetry_emodnet_sign: float = 1.0

    # ========================================================
    # ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
    # ========================================================

    epochs: int = 3
    generations: int = 30
    population_size: int = 24

    elite_count: int = 2
    tournament_size: int = 3

    crossover_probability: float = 0.8
    mutation_probability: float = 0.8
    mutation_fraction: float = 0.25

    random_seed: int = 42
    construction_attempts: int = 1000

    # ========================================================
    # КЕШИ
    # ========================================================

    evaluation_cache_size: int = 128
    weather_cache_size: int = 2048
    ray_cache_size: int = 1024
    profile_cache_size: int = 4096

    # ========================================================
    # ЛОГИРОВАНИЕ И АНИМАЦИЯ
    # ========================================================

    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"

    point_log_every: int = 10
    progress_interval_s: float = 5.0
    log_rotation: str = "50 MB"

    animation_fps: int = 8
    figure_width: int = 1000
    figure_height: int = 800

    @property
    def target_units(self):
        return round(
            2 * self.target_area_m2 / self.cell_side_m**2
        )

    def validate(self):
        positive_names = (
            "target_area_m2",
            "cell_side_m",
            "search_radius_m",
            "evaluation_radius_m",
            "point_spacing_m",
            "tangent_delta_m",
            "probe_m",
            "tolerance_m",
            "max_fetch_m",
            "geodesic_step_m",
            "overwater_factor",
            "rho_water",
            "gravity",
            "reference_depth_m",
            "coastal_depth_m",
            "progress_interval_s",
            "bathymetry_planning_resolution_arcsec",
            "short_profile_pixel_diagonals",
            "nearby_wet_pixel_radius_diagonals",
        )

        for name in positive_names:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name}: требуется конечное число > 0"
                )

        integer_names = (
            "direction_count",
            "max_grid_cells",
            "epochs",
            "generations",
            "population_size",
            "elite_count",
            "tournament_size",
            "construction_attempts",
            "evaluation_cache_size",
            "weather_cache_size",
            "ray_cache_size",
            "profile_cache_size",
            "bathymetry_samples",
            "bathymetry_download_retries",
            "bathymetry_download_timeout_s",
            "bathymetry_max_cells_per_tile",
            "bathymetry_tile_overlap_cells",
            "bathymetry_merge_memory_mb",
            "nearby_wet_pixel_window_radius",
            "point_log_every",
            "animation_fps",
            "figure_width",
            "figure_height",
        )

        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{name}: требуется целое число > 0"
                )

        if not (
            math.isfinite(self.center_lon)
            and -180 <= self.center_lon <= 180
            and math.isfinite(self.center_lat)
            and -89 < self.center_lat < 89
        ):
            raise ValueError("Некорректные координаты центра")

        if self.connection not in {"attached", "island"}:
            raise ValueError("connection: attached | island")

        if self.sea_side not in {"left", "right"}:
            raise ValueError("sea_side: left | right")

        if self.bathymetry_provider not in {
            "auto", "emodnet", "gebco"
        }:
            raise ValueError(
                "bathymetry_provider: auto | emodnet | gebco"
            )

        if 360 % self.direction_count:
            raise ValueError("direction_count должно делить 360")

        if not 1 <= self.elite_count < self.population_size:
            raise ValueError(
                "Нужно 1 <= elite_count < population_size"
            )

        for name in (
            "crossover_probability",
            "mutation_probability",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} должно лежать в [0, 1]")

        for name in (
            "mutation_fraction",
            "breaking_coefficient",
            "min_finite_bathymetry_fraction",
            "min_valid_bathymetry_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} должно лежать в (0, 1]")

        if self.bathymetry_samples < 2:
            raise ValueError("bathymetry_samples должно быть >= 2")

        if not (
            self.tolerance_m
            < self.probe_m
            < self.cell_side_m / 20
        ):
            raise ValueError(
                "Нужно tolerance_m < probe_m < cell_side_m/20"
            )

        if (
            not math.isfinite(self.bathymetry_safety_factor)
            or self.bathymetry_safety_factor < 1
        ):
            raise ValueError(
                "bathymetry_safety_factor должно быть >= 1"
            )

        if (
            not math.isfinite(self.bathymetry_padding_m)
            or self.bathymetry_padding_m < 0
        ):
            raise ValueError("bathymetry_padding_m должно быть >= 0")

        if self.bathymetry_emodnet_sign not in {-1.0, 1.0}:
            raise ValueError(
                "bathymetry_emodnet_sign: -1.0 | 1.0"
            )

        units = 2 * self.target_area_m2 / self.cell_side_m**2

        if not math.isclose(
            units, round(units), rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(
                "Площадь должна быть кратна половине квадрата"
            )

        if self.connection == "island" and round(units) % 2:
            raise ValueError(
                "Площадь острова должна быть кратна полному квадрату"
            )

        if round(units) < (4 if self.connection == "island" else 2):
            raise ValueError("Нужно минимум два смежных квадрата")

        start = (
            date.fromisoformat(self.start_date)
            if self.start_date else None
        )
        end = (
            date.fromisoformat(self.end_date)
            if self.end_date else None
        )

        if start and end and start > end:
            raise ValueError("start_date > end_date")

        for path in (
            self.coastline_path,
            self.weather_path,
            self.land_path,
            self.bathymetry_path,
        ):
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(path)

        if (
            self.bathymetry_required
            and self.bathymetry_path is None
            and not self.bathymetry_auto_download
        ):
            raise ValueError(
                "Батиметрия обязательна: задайте bathymetry_path "
                "или включите bathymetry_auto_download"
            )

        levels = {
            "TRACE", "DEBUG", "INFO", "SUCCESS",
            "WARNING", "ERROR", "CRITICAL",
        }

        for name in ("console_log_level", "file_log_level"):
            if getattr(self, name).upper() not in levels:
                raise ValueError(
                    f"Некорректный уровень {name}"
                )