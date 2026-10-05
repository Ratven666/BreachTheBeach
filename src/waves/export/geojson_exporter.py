from __future__ import annotations

"""Экспорт расчитанных WEI-индексов из SQLite в GeoJSON.

Поддерживаемые режимы:
    points   — каждая точка как Point-объект (lon, lat из wave_points)
    summary  — те же поля + сводная статистика из wave_activity_summary

Компоненты нормали normal_x / normal_y в БД не хранятся. Они
восстанавливаются из азимута нормали (от севера по часовой стрелке):
    normal_x = sin(azimuth)   — составляющая «на восток»
    normal_y = cos(azimuth)   — составляющая «на север»
Азимут в БД округлён до целого градуса, поэтому точность компонент
около 0.5°.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from src.waves.storage import (
    WaveActivityRepository,
    WaveActivitySummaryRow,
    WaveExposureIndexRow,
    WavePointRow,
)


ExportMode = Literal["points", "summary"]

_FLOAT_PRECISION = 6


def _f(value: float | None) -> float | None:
    """Округляет float до читаемой точности, None пропускает."""
    if value is None:
        return None
    return round(float(value), _FLOAT_PRECISION)


def _i(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _normal_components(azimuth_deg: float) -> tuple[float, float]:
    """Единичный вектор нормали (east, north) по азимуту от севера."""
    azimuth_rad = math.radians(float(azimuth_deg))
    return math.sin(azimuth_rad), math.cos(azimuth_rad)


def _index_properties(
    index: WaveExposureIndexRow,
) -> dict:
    return {
        "mean_cwef_wm":       _f(index.mean_cwef_wm),
        "e_storm_mjm":        _f(index.e_storm_mjm),
        "storm_threshold_wm": _f(index.storm_threshold_wm),
        "storm_percentile":   _f(index.storm_percentile),
        "k_dir":              _f(index.k_dir),
        "cv":                 _f(index.cv),
        "n_days":             _i(index.n_days),
        "n_storm_days":       _i(index.n_storm_days),
        "top3_sectors":       index.top3_sectors,
        "r1":                 _i(index.r1),
        "r2":                 _i(index.r2),
        "r3":                 _i(index.r3),
        "r4":                 _i(index.r4),
        "wer":                _f(index.wer),
    }


def _point_properties(
    point: WavePointRow,
) -> dict:
    normal_x, normal_y = _normal_components(point.normal_azimuth_deg)

    return {
        "point_id":           int(point.id),
        "normal_x":           _f(normal_x),
        "normal_y":           _f(normal_y),
        "normal_azimuth_deg": _f(point.normal_azimuth_deg),
    }


def _summary_properties(
    summary: WaveActivitySummaryRow,
) -> dict:
    return {
        "n_active_days":   _i(summary.n_active_days),
        "mean_cwef_wm_s":  _f(summary.mean_cwef_wm),
        "median_cwef_wm":  _f(summary.median_cwef_wm),
        "std_cwef_wm":     _f(summary.std_cwef_wm),
        "p75_cwef_wm":     _f(summary.p75_cwef_wm),
        "p90_cwef_wm":     _f(summary.p90_cwef_wm),
        "p95_cwef_wm":     _f(summary.p95_cwef_wm),
        "p99_cwef_wm":     _f(summary.p99_cwef_wm),
        "max_cwef_wm":     _f(summary.max_cwef_wm),
        "n_storm_days_p90": _i(summary.n_storm_days_p90),
        "total_energy_mjm": _f(summary.total_energy_mjm),
    }


def _make_feature(
    point: WavePointRow,
    index: WaveExposureIndexRow,
    summary: WaveActivitySummaryRow | None,
) -> dict:
    properties: dict = {}
    properties.update(_point_properties(point))
    properties.update(_index_properties(index))

    if summary is not None:
        properties.update(
            _summary_properties(summary)
        )

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [
                round(point.lon, 7),
                round(point.lat, 7),
            ],
        },
        "properties": properties,
    }


@dataclass(frozen=True, slots=True)
class ExportStats:
    feature_count: int
    output_path: Path
    min_wer: float | None
    max_wer: float | None
    mean_wer: float | None


class WaveIndexGeoJSONExporter:
    """Экспортирует индексы волнового воздействия в GeoJSON.

    Пример использования:

        with WaveActivityRepository(db_path) as repo:
            exporter = WaveIndexGeoJSONExporter(repo)
            stats = exporter.export(
                output_path=Path("wave_exposure_index.geojson"),
                mode="summary",
            )
    """

    def __init__(
        self,
        repository: WaveActivityRepository,
    ) -> None:
        self._repo = repository

    def export(
        self,
        output_path: str | Path,
        mode: ExportMode = "summary",
        indent: int | None = None,
        ensure_ascii: bool = False,
    ) -> ExportStats:
        """Записывает GeoJSON на диск.

        Parameters
        ----------
        output_path:
            Путь к выходному файлу.
        mode:
            "points"  — только координаты + WEI-индексы.
            "summary" — то же + сводная статистика CWEF.
        indent:
            Отступ JSON (None = компактный, 2 = читаемый).
        ensure_ascii:
            Если True — экранировать не-ASCII символы.

        Returns
        -------
        ExportStats
        """
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        self._repo.assert_schema_version()

        pairs = self._repo.read_exposure_indices()

        if not pairs:
            raise ValueError(
                "wave_exposure_index is empty. "
                "Run db_pypeline/5_2_wave_exposure_to_db.py first."
            )

        logger.info(
            "Exporting {} points to GeoJSON (mode={})",
            len(pairs),
            mode,
        )

        summaries: dict[int, WaveActivitySummaryRow] = {}

        if mode == "summary":
            for row in self._repo.read_summaries():
                summaries[row.point_id] = row

        features: list[dict] = []

        for point, index in pairs:
            summary = (
                summaries.get(point.id)
                if mode == "summary"
                else None
            )
            features.append(
                _make_feature(point, index, summary)
            )

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84",
                },
            },
            "features": features,
        }

        tmp = output.with_suffix(".tmp.geojson")

        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(
                    geojson,
                    fh,
                    ensure_ascii=ensure_ascii,
                    indent=indent,
                )

            tmp.replace(output)

        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

        wer_values = [
            float(idx.wer)
            for _, idx in pairs
            if idx.wer is not None
        ]

        stats = ExportStats(
            feature_count=len(features),
            output_path=output,
            min_wer=min(wer_values) if wer_values else None,
            max_wer=max(wer_values) if wer_values else None,
            mean_wer=(
                sum(wer_values) / len(wer_values)
                if wer_values
                else None
            ),
        )

        logger.success(
            "GeoJSON exported: features={}, "
            "WER=[{}, {}] mean={}, path={}",
            stats.feature_count,
            (
                f"{stats.min_wer:.3f}"
                if stats.min_wer is not None
                else "—"
            ),
            (
                f"{stats.max_wer:.3f}"
                if stats.max_wer is not None
                else "—"
            ),
            (
                f"{stats.mean_wer:.3f}"
                if stats.mean_wer is not None
                else "—"
            ),
            stats.output_path,
        )

        return stats