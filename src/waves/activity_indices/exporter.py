from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .constants import EPOCH, GEOJSON_MODE_TRENDS, GEOJSON_MODES
from .repository import IndexRepository


@dataclass(frozen=True, slots=True)
class ExportStats:
    feature_count: int
    output_path: Path


def _feature(row: dict) -> dict:
    lon, lat = row.pop("lon"), row.pop("lat")
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": row,
    }


def export_geojson(
    index_db: Path,
    output_path: Path,
    *,
    mode: str,
    indent: int | None,
) -> ExportStats:
    if mode not in GEOJSON_MODES:
        raise ValueError(f"Unknown mode {mode!r}; use one of {GEOJSON_MODES}")

    if mode == GEOJSON_MODE_TRENDS:
        sql = (
            "SELECT p.point_id, p.lon, p.lat, p.normal_azimuth_deg, t.* "
            "FROM points p JOIN point_trends t USING(point_id) ORDER BY p.point_id"
        )
    else:
        sql = (
            "SELECT p.point_id, p.lon, p.lat, p.normal_azimuth_deg, "
            "q.period_id, q.start_day, q.end_day, q.n_days, q.is_complete, q.mid_year, "
            "i.n_active_days, i.normal_mean_wm, i.alongshore_net_mean_wm, "
            "i.alongshore_abs_mean_wm, i.alongshore_pos_mean_wm, i.alongshore_neg_mean_wm, "
            "i.d_normal_wm, i.d_normal_pct, i.d_alongshore_abs_wm, "
            "i.d_alongshore_abs_pct, i.d_alongshore_net_wm "
            "FROM point_period_indices i JOIN points p USING(point_id) "
            "JOIN periods q USING(period_id) ORDER BY q.period_id, p.point_id"
        )

    features = []
    with IndexRepository(index_db, readonly=True) as repo:
        for row in repo.iter_query(sql):
            d = dict(row)
            if "start_day" in d:
                d["start_date"] = (EPOCH + timedelta(days=d["start_day"])).isoformat()
                d["end_date"] = (EPOCH + timedelta(days=d["end_day"])).isoformat()
            features.append(_feature(d))

    collection = {
        "type": "FeatureCollection",
        "name": f"wave_activity_indices_{mode}",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collection, ensure_ascii=False, indent=indent, allow_nan=False),
        encoding="utf-8",
    )
    return ExportStats(len(features), output_path)
