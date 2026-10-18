from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from .repository import LocalMoranRepository


@dataclass(frozen=True, slots=True)
class LocalMoranExportStats:
    run_id: int
    feature_count: int
    output_path: Path


def export_local_moran_geojson(
    db_path: Path,
    output_path: Path,
    *,
    run_id: int | None = None,
) -> LocalMoranExportStats:
    repo = LocalMoranRepository(db_path)
    repo.initialize()

    selected_run_id = repo.get_last_run_id() if run_id is None else int(run_id)
    if selected_run_id is None:
        raise ValueError("No Local Moran runs found in database")

    run = repo.get_run(selected_run_id)
    if run is None:
        raise ValueError(f"Run id={selected_run_id} not found")

    rows = repo.read_export_rows(selected_run_id)
    if not rows:
        raise ValueError(f"No export rows found for run id={selected_run_id}")

    grouped: dict[int, dict] = {}

    for row in rows:
        point_id = int(row["point_id"])

        feature = grouped.setdefault(
            point_id,
            {
                "run_id": selected_run_id,
                "point_id": point_id,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "normal_azimuth_deg": int(row["normal_azimuth_deg"]),
                "segment_id": str(row["segment_id"]),
                "order_in_segment": int(row["order_in_segment"]),
                "chainage_m": float(row["chainage_m"]),
                "segment_length_m": float(row["segment_length_m"]),
                "snap_distance_m": float(row["snap_distance_m"]),
                "snap_ok": int(row["snap_ok"]),
                "is_closed": int(row["is_closed"]),
                "alongshore_metric": str(run.alongshore_metric),
            },
        )

        prefix = "normal" if row["metric"] == "normal_mean_wm" else "alongshore"

        for source_name, target_name in (
            ("value", f"{prefix}_value"),
            ("n_window", f"{prefix}_n_window"),
            ("n_neighbors", f"{prefix}_n_neighbors"),
            ("window_mean", f"{prefix}_window_mean"),
            ("window_sd", f"{prefix}_window_sd"),
            ("z_value", f"{prefix}_z_value"),
            ("spatial_lag", f"{prefix}_spatial_lag"),
            ("local_i", f"{prefix}_local_i"),
            ("permutation_expected_i", f"{prefix}_permutation_expected_i"),
            ("permutation_z", f"{prefix}_permutation_z"),
            ("p_sim", f"{prefix}_p_sim"),
            ("q_fdr_bh", f"{prefix}_q_fdr_bh"),
            ("quadrant", f"{prefix}_quadrant"),
            ("cluster", f"{prefix}_cluster"),
            ("cluster_fdr", f"{prefix}_cluster_fdr"),
            ("status", f"{prefix}_status"),
        ):
            feature[target_name] = row[source_name]

    feature_rows = list(grouped.values())

    gdf = gpd.GeoDataFrame(
        feature_rows,
        geometry=gpd.points_from_xy(
            [row["lon"] for row in feature_rows],
            [row["lat"] for row in feature_rows],
        ),
        crs="EPSG:4326",
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    gdf.to_file(output_path, driver="GeoJSON")

    return LocalMoranExportStats(
        run_id=selected_run_id,
        feature_count=len(gdf),
        output_path=output_path,
    )
