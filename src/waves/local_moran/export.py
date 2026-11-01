from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd


@dataclass(frozen=True, slots=True)
class MoranExportStats:
    run_id: int
    feature_count: int
    output_path: Path


def _result_properties(
    row: sqlite3.Row | None,
    prefix: str,
) -> dict:
    fields = (
        "value",
        "n_window",
        "n_neighbors",
        "window_mean",
        "window_sd",
        "z_value",
        "spatial_lag",
        "local_i",
        "permutation_expected_i",
        "permutation_z",
        "p_sim",
        "q_fdr_bh",
        "quadrant",
        "cluster",
        "cluster_fdr",
        "status",
    )

    if row is None:
        return {
            f"{prefix}_{field}": None
            for field in fields
        }

    return {
        f"{prefix}_{field}": row[field]
        for field in fields
    }


def export_local_moran_geojson(
    index_db: str | Path,
    output_geojson: str | Path,
    *,
    run_id: int | None = None,
) -> MoranExportStats:
    database_path = Path(index_db).expanduser().resolve()
    output_path = Path(output_geojson).expanduser().resolve()

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Index database not found: {database_path}"
        )

    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    try:
        if run_id is None:
            run_row = connection.execute(
                """
                SELECT *
                FROM local_moran_runs
                ORDER BY run_id DESC
                LIMIT 1
                """
            ).fetchone()
        else:
            run_row = connection.execute(
                """
                SELECT *
                FROM local_moran_runs
                WHERE run_id = ?
                """,
                (int(run_id),),
            ).fetchone()

        if run_row is None:
            raise ValueError(
                "No matching local Moran run found"
            )

        selected_run_id = int(run_row["run_id"])
        alongshore_metric = str(run_row["alongshore_metric"])

        mapping_rows = connection.execute(
            """
            SELECT
                p.point_id,
                p.lon,
                p.lat,
                p.normal_azimuth_deg,
                o.segment_id,
                o.order_in_segment,
                o.chainage_m,
                o.segment_length_m,
                o.snap_distance_m,
                o.is_closed
            FROM local_moran_point_order AS o
            JOIN points AS p
                ON p.point_id = o.point_id
            WHERE o.run_id = ?
            ORDER BY
                o.segment_id,
                o.order_in_segment,
                p.point_id
            """,
            (selected_run_id,),
        ).fetchall()

        if not mapping_rows:
            raise ValueError(
                f"No point-order rows found for run_id={selected_run_id}"
            )

        result_rows = connection.execute(
            """
            SELECT *
            FROM local_moran_results
            WHERE run_id = ?
            """,
            (selected_run_id,),
        ).fetchall()

        lookup = {
            (int(row["point_id"]), str(row["metric"])): row
            for row in result_rows
        }

        records: list[dict] = []

        for row in mapping_rows:
            point_id = int(row["point_id"])

            normal = lookup.get(
                (point_id, "normal_mean_wm")
            )
            alongshore = lookup.get(
                (point_id, alongshore_metric)
            )

            properties = {
                "run_id": selected_run_id,
                "point_id": point_id,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "normal_azimuth_deg": row["normal_azimuth_deg"],
                "segment_id": str(row["segment_id"]),
                "order_in_segment": int(row["order_in_segment"]),
                "chainage_m": float(row["chainage_m"]),
                "segment_length_m": float(row["segment_length_m"]),
                "snap_distance_m": float(row["snap_distance_m"]),
                "is_closed": int(row["is_closed"]),
                "snap_ok": 1,
                "alongshore_metric": alongshore_metric,
            }

            properties.update(
                _result_properties(normal, "normal")
            )
            properties.update(
                _result_properties(alongshore, "alongshore")
            )

            records.append(properties)

        gdf = gpd.GeoDataFrame(
            records,
            geometry=gpd.points_from_xy(
                [row["lon"] for row in records],
                [row["lat"] for row in records],
            ),
            crs="EPSG:4326",
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        gdf.to_file(output_path, driver="GeoJSON")

        return MoranExportStats(
            run_id=selected_run_id,
            feature_count=len(gdf),
            output_path=output_path,
        )

    finally:
        connection.close()