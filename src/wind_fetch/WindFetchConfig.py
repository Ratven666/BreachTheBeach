# src/wind_fetch/WindFetchConfig.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class WindFetchConfig:
    """
    Конфигурация расчёта fetch.

    Все расстояния задаются в метрах.
    Все углы — azimuth/bearing:
    0 = north, 90 = east, 180 = south, 270 = west.
    """

    # ── Выходные файлы (оригинальные поля) ────────────────────────────────
    output_dir: Path = Path("output")

    multi_output_csv_name: str = "multi_direction_fetch.csv"
    multi_output_points_name: str = "multi_direction_source_points.geojson"
    multi_output_start_points_name: str = "multi_direction_start_points.geojson"
    multi_output_offset_segments_name: str = "multi_direction_offset_segments.geojson"
    multi_output_rays_name: str = "multi_direction_rays.geojson"
    multi_output_hit_points_name: str = "multi_direction_hit_points.geojson"
    multi_output_split_dirname: str = "multi_direction_split"

    # Алиасы для WindFetchCalculator (однонаправленный режим)
    output_csv_name: str = "wind_fetch_results.csv"
    output_geojson_name: str = "wind_fetch_rays.geojson"

    # ── Геометрия трассировки (оригинальные поля) ──────────────────────────
    default_offset_m: float = 1.0
    default_fetch_m: float = 100_000.0

    coastal_exclusion_m: float = 1.0
    normal_azimuth_field: str = "normal_azimuth_deg"

    use_make_valid: bool = True
    precision_grid_m: float = 0.05

    azimuths_deg: list[float] = field(
        default_factory=lambda: [float(v) for v in range(360)]
    )

    # ── НОВЫЕ поля (добавлены для WindFetchParallelRunner) ─────────────────
    # Шаг геодезического луча в метрах
    geodesic_step_m: float = 1_000.0
    # Максимальное число сегментов на один луч
    max_segments_per_ray: int = 200

    # ── Параллельный запуск ────────────────────────────────────────────────
    n_workers: int = 4
    chunk_size: int = 50

    # ── Сектор суши (для параллельного и многонаправленного режимов) ───────
    half_land_sector_deg: float = 90.0
