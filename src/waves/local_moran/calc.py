from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
from loguru import logger
from pyproj import CRS
from shapely.geometry import GeometryCollection, LineString, MultiLineString


ALONGSHORE_MODE_TO_COLUMN = {
    "abs": "alongshore_abs_mean_wm",
    "net": "alongshore_net_mean_wm",
}

_METRIC_CODES = {
    "normal_mean_wm": 1,
    "alongshore_abs_mean_wm": 2,
    "alongshore_net_mean_wm": 3,
}


@dataclass(frozen=True, slots=True)
class PreparedGeometry:
    point_rows: list[dict]
    coastline_crs: str
    metric_crs: str
    n_total: int
    n_snap_ok: int
    n_snap_bad: int
    mean_snap_distance_m: float
    max_snap_distance_found_m: float


def resolve_alongshore_metric(alongshore_mode: str) -> str:
    mode = str(alongshore_mode).strip().lower()

    if mode not in ALONGSHORE_MODE_TO_COLUMN:
        raise ValueError(
            f"Unsupported alongshore mode: {alongshore_mode!r}. "
            f"Expected one of {sorted(ALONGSHORE_MODE_TO_COLUMN)}"
        )

    return ALONGSHORE_MODE_TO_COLUMN[mode]


def _iter_line_parts(geometry) -> Iterator[LineString]:
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, LineString):
        if geometry.length > 0.0:
            yield geometry
        return

    if isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            if not part.is_empty and part.length > 0.0:
                yield part
        return

    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _iter_line_parts(item)
        return


def _load_segments(
    coastline_geojson: Path,
    metric_crs: str | None,
) -> tuple[gpd.GeoDataFrame, CRS, str]:
    coastline = gpd.read_file(coastline_geojson)

    if coastline.empty:
        raise ValueError(f"Coastline GeoJSON is empty: {coastline_geojson}")

    if coastline.crs is None:
        coastline = coastline.set_crs("EPSG:4326")

    coastline_crs = CRS.from_user_input(coastline.crs).to_string()

    records: list[dict] = []
    segment_rank = 0

    for feature_index, geometry in enumerate(coastline.geometry):
        for part_index, part in enumerate(_iter_line_parts(geometry)):
            records.append(
                {
                    "segment_id": (
                        f"feature_{feature_index:06d}"
                        f"_part_{part_index:04d}"
                    ),
                    "segment_rank": segment_rank,
                    "geometry": part,
                }
            )
            segment_rank += 1

    if not records:
        raise ValueError(
            "Coastline GeoJSON does not contain non-empty "
            "LineString or MultiLineString geometries"
        )

    segments = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=coastline.crs,
    )

    if metric_crs is None:
        estimated = segments.to_crs("EPSG:4326").estimate_utm_crs()
        if estimated is None:
            raise ValueError(
                "Unable to estimate projected CRS from coastline. "
                "Pass metric_crs explicitly."
            )
        target_crs = CRS.from_user_input(estimated)
    else:
        target_crs = CRS.from_user_input(metric_crs)

    if target_crs.is_geographic:
        raise ValueError(
            f"metric_crs must be projected in linear units, got {target_crs}"
        )

    return segments.to_crs(target_crs), target_crs, coastline_crs


def assign_points_to_coastline(
    source_rows: list[dict],
    coastline_geojson: Path,
    metric_crs: str | None,
    max_snap_distance_m: float | None,
    strict_snap: bool,
) -> PreparedGeometry:
    segments, target_crs, coastline_crs = _load_segments(
        Path(coastline_geojson),
        metric_crs,
    )

    points = gpd.GeoDataFrame(
        source_rows,
        geometry=gpd.points_from_xy(
            [float(row["lon"]) for row in source_rows],
            [float(row["lat"]) for row in source_rows],
        ),
        crs="EPSG:4326",
    ).to_crs(target_crs)

    joined = gpd.sjoin_nearest(
        points,
        segments[["segment_id", "segment_rank", "geometry"]],
        how="left",
        distance_col="snap_distance_m",
    )

    if joined["segment_id"].isna().any():
        missing_ids = (
            joined.loc[joined["segment_id"].isna(), "point_id"]
            .astype(int)
            .tolist()
        )
        raise ValueError(
            "Some points could not be assigned to a coastline segment. "
            f"Sample point_ids={missing_ids[:10]}"
        )

    joined = (
        joined.sort_values(
            ["point_id", "snap_distance_m", "segment_rank"]
        )
        .drop_duplicates(subset=["point_id"], keep="first")
        .copy()
    )

    segment_lookup = {
        str(row.segment_id): row.geometry
        for row in segments.itertuples()
    }

    point_rows: list[dict] = []

    for row in joined.itertuples():
        point_id = int(row.point_id)
        segment_id = str(row.segment_id)
        line = segment_lookup[segment_id]

        chainage_m = float(line.project(row.geometry))
        segment_length_m = float(line.length)
        snap_distance_m = float(row.snap_distance_m)

        snap_ok = True
        if max_snap_distance_m is not None:
            snap_ok = snap_distance_m <= float(max_snap_distance_m) + 1.0e-12

        point_rows.append(
            {
                "point_id": point_id,
                "lon": float(row.lon),
                "lat": float(row.lat),
                "normal_azimuth_deg": (
                    None
                    if row.normal_azimuth_deg is None
                    else float(row.normal_azimuth_deg)
                ),
                "normal_mean_wm": (
                    None
                    if row.normal_mean_wm is None
                    else float(row.normal_mean_wm)
                ),
                "alongshore_mean_wm": (
                    None
                    if row.alongshore_mean_wm is None
                    else float(row.alongshore_mean_wm)
                ),
                "segment_id": segment_id,
                "segment_rank": int(row.segment_rank),
                "chainage_m": chainage_m,
                "segment_length_m": segment_length_m,
                "snap_distance_m": snap_distance_m,
                "snap_ok": bool(snap_ok),
                "is_closed": bool(line.is_ring),
            }
        )

    point_rows.sort(
        key=lambda item: (
            item["segment_id"],
            float(item["chainage_m"]),
            int(item["point_id"]),
        )
    )

    current_segment_id: str | None = None
    order_in_segment = 0

    for row in point_rows:
        if row["segment_id"] != current_segment_id:
            current_segment_id = str(row["segment_id"])
            order_in_segment = 0

        row["order_in_segment"] = order_in_segment
        order_in_segment += 1

    n_total = len(point_rows)
    n_snap_ok = sum(1 for row in point_rows if bool(row["snap_ok"]))
    n_snap_bad = n_total - n_snap_ok

    snap_distances = np.asarray(
        [float(row["snap_distance_m"]) for row in point_rows],
        dtype=np.float64,
    )

    mean_snap_distance_m = (
        float(np.mean(snap_distances)) if snap_distances.size else 0.0
    )
    max_snap_distance_found_m = (
        float(np.max(snap_distances)) if snap_distances.size else 0.0
    )

    if max_snap_distance_m is not None and n_snap_bad > 0:
        sample_bad = [
            (
                int(row["point_id"]),
                round(float(row["snap_distance_m"]), 3),
            )
            for row in point_rows
            if not bool(row["snap_ok"])
        ][:10]

        logger.warning(
            "Some points exceed max_snap_distance_m={}. Count={}; sample={}",
            max_snap_distance_m,
            n_snap_bad,
            sample_bad,
        )

        if strict_snap:
            raise ValueError(
                "Found points farther than max_snap_distance_m="
                f"{max_snap_distance_m}. Sample={sample_bad}"
            )

    return PreparedGeometry(
        point_rows=point_rows,
        coastline_crs=coastline_crs,
        metric_crs=target_crs.to_string(),
        n_total=n_total,
        n_snap_ok=n_snap_ok,
        n_snap_bad=n_snap_bad,
        mean_snap_distance_m=mean_snap_distance_m,
        max_snap_distance_found_m=max_snap_distance_found_m,
    )


def filter_snap_ok_rows(point_rows: list[dict]) -> list[dict]:
    return [row for row in point_rows if bool(row.get("snap_ok", True))]


def alongshore_distance(
    *,
    chainages: np.ndarray,
    focal_chainage: float,
    segment_length_m: float,
    is_closed: bool,
) -> np.ndarray:
    distances = np.abs(chainages - float(focal_chainage))

    if is_closed and segment_length_m > 0.0:
        distances = np.minimum(
            distances,
            np.maximum(0.0, segment_length_m - distances),
        )

    return distances


def _quadrant(z_value: float, spatial_lag: float) -> str:
    eps = 1.0e-14

    if abs(z_value) <= eps or abs(spatial_lag) <= eps:
        return "UNDEFINED"

    if z_value > 0.0 and spatial_lag > 0.0:
        return "HH"

    if z_value < 0.0 and spatial_lag < 0.0:
        return "LL"

    if z_value > 0.0 and spatial_lag < 0.0:
        return "HL"

    return "LH"


def _empty_result(
    point_id: int,
    metric: str,
    value: float | None,
    n_window: int,
    n_neighbors: int,
    status: str,
) -> dict:
    return {
        "point_id": int(point_id),
        "metric": str(metric),
        "value": value,
        "n_window": int(n_window),
        "n_neighbors": int(n_neighbors),
        "window_mean": None,
        "window_sd": None,
        "z_value": None,
        "spatial_lag": None,
        "local_i": None,
        "permutation_expected_i": None,
        "permutation_z": None,
        "p_sim": None,
        "q_fdr_bh": None,
        "quadrant": None,
        "cluster": status,
        "cluster_fdr": status,
        "status": status,
    }


def _seed_sequence(seed: int, point_id: int, metric: str) -> np.random.SeedSequence:
    if metric not in _METRIC_CODES:
        raise ValueError(f"Unsupported metric code mapping for {metric!r}")

    unsigned_point_id = int(point_id) & ((1 << 64) - 1)

    return np.random.SeedSequence(
        [
            int(seed),
            unsigned_point_id & 0xFFFFFFFF,
            (unsigned_point_id >> 32) & 0xFFFFFFFF,
            _METRIC_CODES[metric],
        ]
    )


def compute_local_moran_for_metric(
    point_rows: list[dict],
    *,
    metric: str,
    value_key: str,
    radius_m: float,
    window_width_m: float,
    weight_mode: str,
    permutations: int,
    seed: int,
    alpha_raw: float,
) -> list[dict]:
    if radius_m <= 0.0:
        raise ValueError("radius_m must be > 0")
    if window_width_m <= 0.0:
        raise ValueError("window_width_m must be > 0")
    if window_width_m + 1.0e-9 < 2.0 * radius_m:
        raise ValueError("window_width_m must be at least 2 * radius_m")
    if permutations < 99:
        raise ValueError("permutations must be >= 99")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if not 0.0 < alpha_raw < 1.0:
        raise ValueError("alpha_raw must be in (0, 1)")
    if weight_mode not in {"binary", "inverse_distance"}:
        raise ValueError("weight_mode must be 'binary' or 'inverse_distance'")

    half_window = 0.5 * float(window_width_m)
    results: list[dict] = []

    segment_ids = sorted({str(row["segment_id"]) for row in point_rows})

    for segment_id in segment_ids:
        rows = [
            row for row in point_rows
            if str(row["segment_id"]) == segment_id
        ]

        rows.sort(
            key=lambda item: (
                float(item["chainage_m"]),
                int(item["point_id"]),
            )
        )

        point_ids = [int(row["point_id"]) for row in rows]
        chainages = np.asarray(
            [float(row["chainage_m"]) for row in rows],
            dtype=np.float64,
        )
        values = np.asarray(
            [
                np.nan
                if row.get(value_key) is None
                else float(row[value_key])
                for row in rows
            ],
            dtype=np.float64,
        )

        indices = np.arange(len(rows), dtype=np.int64)
        segment_length_m = float(rows[0]["segment_length_m"])
        is_closed = bool(rows[0]["is_closed"])

        for focal_index, point_id in enumerate(point_ids):
            distances = alongshore_distance(
                chainages=chainages,
                focal_chainage=float(chainages[focal_index]),
                segment_length_m=segment_length_m,
                is_closed=is_closed,
            )

            finite = np.isfinite(values)

            window_mask = finite & (distances <= half_window + 1.0e-9)
            neighbor_mask = (
                window_mask
                & (indices != focal_index)
                & (distances <= radius_m + 1.0e-9)
            )

            n_window = int(np.count_nonzero(window_mask))
            n_neighbors = int(np.count_nonzero(neighbor_mask))

            focal_value = (
                None
                if not np.isfinite(values[focal_index])
                else float(values[focal_index])
            )

            if focal_value is None:
                results.append(
                    _empty_result(
                        point_id,
                        metric,
                        None,
                        n_window,
                        n_neighbors,
                        "MISSING_VALUE",
                    )
                )
                continue

            if n_window < 3:
                results.append(
                    _empty_result(
                        point_id,
                        metric,
                        focal_value,
                        n_window,
                        n_neighbors,
                        "INSUFFICIENT_WINDOW",
                    )
                )
                continue

            if n_neighbors < 1:
                results.append(
                    _empty_result(
                        point_id,
                        metric,
                        focal_value,
                        n_window,
                        0,
                        "NO_NEIGHBORS",
                    )
                )
                continue

            window_values = values[window_mask]
            window_mean = float(np.mean(window_values))
            window_sd = float(
                np.sqrt(np.mean((window_values - window_mean) ** 2))
            )

            if not np.isfinite(window_sd) or window_sd <= 1.0e-14:
                row = _empty_result(
                    point_id,
                    metric,
                    focal_value,
                    n_window,
                    n_neighbors,
                    "NO_VARIATION",
                )
                row["window_mean"] = window_mean
                row["window_sd"] = window_sd
                results.append(row)
                continue

            z = (values - window_mean) / window_sd
            z_value = float(z[focal_index])

            neighbor_indices = indices[neighbor_mask]
            neighbor_distances = distances[neighbor_mask]

            if weight_mode == "binary":
                weights = np.ones(neighbor_indices.size, dtype=np.float64)
            else:
                weights = 1.0 / np.maximum(neighbor_distances, 1.0e-6)

            weights = weights / float(np.sum(weights))

            spatial_lag = float(np.dot(weights, z[neighbor_indices]))
            local_i = float(z_value * spatial_lag)

            candidate_mask = window_mask & (indices != focal_index)
            candidate_values = z[candidate_mask]

            if candidate_values.size < neighbor_indices.size:
                results.append(
                    _empty_result(
                        point_id,
                        metric,
                        focal_value,
                        n_window,
                        n_neighbors,
                        "INSUFFICIENT_CANDIDATES",
                    )
                )
                continue

            rng = np.random.default_rng(
                _seed_sequence(seed, point_id, metric)
            )

            simulated = np.empty(permutations, dtype=np.float64)

            for simulation_index in range(permutations):
                chosen = rng.choice(
                    candidate_values,
                    size=neighbor_indices.size,
                    replace=False,
                )
                simulated[simulation_index] = float(
                    z_value * np.dot(weights, chosen)
                )

            expected_i = float(np.mean(simulated))
            sim_sd = (
                float(np.std(simulated, ddof=1))
                if permutations > 1
                else 0.0
            )

            permutation_z = (
                None
                if sim_sd <= 1.0e-14
                else float((local_i - expected_i) / sim_sd)
            )

            observed_abs = abs(local_i - expected_i)
            simulated_abs = np.abs(simulated - expected_i)
            p_sim = float(
                (
                    1
                    + np.count_nonzero(
                        simulated_abs >= observed_abs - 1.0e-15
                    )
                )
                / (permutations + 1)
            )

            quadrant = _quadrant(z_value, spatial_lag)
            cluster = (
                quadrant
                if p_sim <= alpha_raw and quadrant != "UNDEFINED"
                else "NS"
            )

            results.append(
                {
                    "point_id": point_id,
                    "metric": metric,
                    "value": focal_value,
                    "n_window": n_window,
                    "n_neighbors": n_neighbors,
                    "window_mean": window_mean,
                    "window_sd": window_sd,
                    "z_value": z_value,
                    "spatial_lag": spatial_lag,
                    "local_i": local_i,
                    "permutation_expected_i": expected_i,
                    "permutation_z": permutation_z,
                    "p_sim": p_sim,
                    "q_fdr_bh": None,
                    "quadrant": quadrant,
                    "cluster": cluster,
                    "cluster_fdr": None,
                    "status": "OK",
                }
            )

    return results


def apply_bh_fdr(
    rows: list[dict],
    alpha_fdr: float,
) -> None:
    if not 0.0 < alpha_fdr < 1.0:
        raise ValueError("alpha_fdr must be in (0, 1)")

    valid_indices = [
        index
        for index, row in enumerate(rows)
        if row["status"] == "OK" and row["p_sim"] is not None
    ]

    if not valid_indices:
        return

    p_values = np.asarray(
        [rows[index]["p_sim"] for index in valid_indices],
        dtype=np.float64,
    )

    order = np.argsort(p_values)
    p_sorted = p_values[order]
    n = p_sorted.size

    q_sorted = p_sorted * n / np.arange(1, n + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_values = np.empty_like(q_sorted)
    q_values[order] = q_sorted

    for row_index, q_value in zip(valid_indices, q_values):
        row = rows[row_index]
        row["q_fdr_bh"] = float(q_value)
        row["cluster_fdr"] = (
            row["quadrant"]
            if q_value <= alpha_fdr and row["quadrant"] != "UNDEFINED"
            else "NS"
        )
