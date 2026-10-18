from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    LocalMoranPointOrderModel,
    LocalMoranResultModel,
    LocalMoranRunModel,
)


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalMoranRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.engine: Engine = create_engine(
            f"sqlite:///{self.db_path}",
            future=True,
        )
        event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            future=True,
            expire_on_commit=False,
        )

    def session(self) -> Session:
        return self.SessionLocal()

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        self._migrate_existing_schema()

    def _get_column_names(self, table_name: str) -> set[str]:
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())

        if table_name not in table_names:
            return set()

        return {
            str(column["name"])
            for column in inspector.get_columns(table_name)
        }

    def _add_column_if_missing(
        self,
        table_name: str,
        column_name: str,
        ddl_sql: str,
    ) -> None:
        existing = self._get_column_names(table_name)
        if column_name in existing:
            return

        with self.engine.begin() as connection:
            connection.execute(text(ddl_sql))

    def _migrate_existing_schema(self) -> None:
        table_names = set(inspect(self.engine).get_table_names())

        if "local_moran_runs" in table_names:
            self._add_column_if_missing(
                "local_moran_runs",
                "strict_snap",
                """
                ALTER TABLE local_moran_runs
                ADD COLUMN strict_snap INTEGER NOT NULL DEFAULT 0
                """,
            )
            self._add_column_if_missing(
                "local_moran_runs",
                "n_points_total",
                """
                ALTER TABLE local_moran_runs
                ADD COLUMN n_points_total INTEGER NOT NULL DEFAULT 0
                """,
            )
            self._add_column_if_missing(
                "local_moran_runs",
                "n_points_snap_ok",
                """
                ALTER TABLE local_moran_runs
                ADD COLUMN n_points_snap_ok INTEGER NOT NULL DEFAULT 0
                """,
            )
            self._add_column_if_missing(
                "local_moran_runs",
                "alpha_raw",
                """
                ALTER TABLE local_moran_runs
                ADD COLUMN alpha_raw REAL NOT NULL DEFAULT 0.05
                """,
            )
            self._add_column_if_missing(
                "local_moran_runs",
                "alpha_fdr",
                """
                ALTER TABLE local_moran_runs
                ADD COLUMN alpha_fdr REAL NOT NULL DEFAULT 0.05
                """,
            )

            existing = self._get_column_names("local_moran_runs")

            if "alpha" in existing:
                with self.engine.begin() as connection:
                    connection.execute(
                        text(
                            """
                            UPDATE local_moran_runs
                            SET alpha_raw = COALESCE(alpha_raw, alpha, 0.05)
                            WHERE alpha_raw IS NULL OR alpha_raw = 0
                            """
                        )
                    )
                    connection.execute(
                        text(
                            """
                            UPDATE local_moran_runs
                            SET alpha_fdr = COALESCE(alpha_fdr, alpha, 0.05)
                            WHERE alpha_fdr IS NULL OR alpha_fdr = 0
                            """
                        )
                    )

            if "n_points" in existing:
                with self.engine.begin() as connection:
                    connection.execute(
                        text(
                            """
                            UPDATE local_moran_runs
                            SET n_points_total = COALESCE(n_points_total, n_points, 0)
                            WHERE n_points_total = 0
                            """
                        )
                    )
                    connection.execute(
                        text(
                            """
                            UPDATE local_moran_runs
                            SET n_points_snap_ok = COALESCE(n_points_snap_ok, n_points_total, 0)
                            WHERE n_points_snap_ok = 0
                            """
                        )
                    )

        if "local_moran_point_order" in table_names:
            self._add_column_if_missing(
                "local_moran_point_order",
                "snap_ok",
                """
                ALTER TABLE local_moran_point_order
                ADD COLUMN snap_ok INTEGER NOT NULL DEFAULT 1
                """,
            )

    def read_source_points_and_trends(self, alongshore_metric: str) -> list[dict]:
        if alongshore_metric not in {
            "alongshore_abs_mean_wm",
            "alongshore_net_mean_wm",
        }:
            raise ValueError(f"Unsupported alongshore metric: {alongshore_metric}")

        sql = text(
            f"""
            SELECT
                p.point_id AS point_id,
                p.lon AS lon,
                p.lat AS lat,
                p.normal_azimuth_deg AS normal_azimuth_deg,
                t.normal_mean_wm AS normal_mean_wm,
                t.{alongshore_metric} AS alongshore_mean_wm
            FROM points AS p
            LEFT JOIN point_trends AS t
                ON t.point_id = p.point_id
            ORDER BY p.point_id
            """
        )

        with self.session() as session:
            rows = session.execute(sql).mappings().all()

        if not rows:
            raise ValueError("No source points found in database")

        return [dict(row) for row in rows]

    def insert_run(
        self,
        *,
        created_utc: str,
        coastline_path: str,
        coastline_sha256: str,
        coastline_crs: str,
        metric_crs: str,
        radius_m: float,
        window_width_m: float,
        max_snap_distance_m: float | None,
        weight_mode: str,
        permutations: int,
        random_seed: int,
        alpha_raw: float,
        alpha_fdr: float,
        alongshore_metric: str,
        strict_snap: bool,
        n_points_total: int,
        n_points_snap_ok: int,
        n_results: int,
    ) -> int:
        with self.session() as session:
            model = LocalMoranRunModel(
                created_utc=created_utc,
                coastline_path=coastline_path,
                coastline_sha256=coastline_sha256,
                coastline_crs=coastline_crs,
                metric_crs=metric_crs,
                radius_m=radius_m,
                window_width_m=window_width_m,
                max_snap_distance_m=max_snap_distance_m,
                weight_mode=weight_mode,
                permutations=permutations,
                random_seed=random_seed,
                alpha_raw=alpha_raw,
                alpha_fdr=alpha_fdr,
                alongshore_metric=alongshore_metric,
                strict_snap=int(strict_snap),
                n_points_total=n_points_total,
                n_points_snap_ok=n_points_snap_ok,
                n_results=n_results,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return int(model.run_id)

    def insert_point_order(self, run_id: int, rows: list[dict]) -> None:
        payload = [
            LocalMoranPointOrderModel(
                run_id=run_id,
                point_id=int(row["point_id"]),
                segment_id=str(row["segment_id"]),
                order_in_segment=int(row["order_in_segment"]),
                chainage_m=float(row["chainage_m"]),
                segment_length_m=float(row["segment_length_m"]),
                snap_distance_m=float(row["snap_distance_m"]),
                snap_ok=int(bool(row["snap_ok"])),
                is_closed=int(bool(row["is_closed"])),
            )
            for row in rows
        ]

        with self.session() as session:
            session.add_all(payload)
            session.commit()

    def insert_results(self, run_id: int, rows: list[dict]) -> None:
        payload = [
            LocalMoranResultModel(
                run_id=run_id,
                point_id=int(row["point_id"]),
                metric=str(row["metric"]),
                value=row["value"],
                n_window=int(row["n_window"]),
                n_neighbors=int(row["n_neighbors"]),
                window_mean=row["window_mean"],
                window_sd=row["window_sd"],
                z_value=row["z_value"],
                spatial_lag=row["spatial_lag"],
                local_i=row["local_i"],
                permutation_expected_i=row["permutation_expected_i"],
                permutation_z=row["permutation_z"],
                p_sim=row["p_sim"],
                q_fdr_bh=row["q_fdr_bh"],
                quadrant=row["quadrant"],
                cluster=row["cluster"],
                cluster_fdr=row["cluster_fdr"],
                status=str(row["status"]),
            )
            for row in rows
        ]

        with self.session() as session:
            session.add_all(payload)
            session.commit()

    def get_last_run_id(self) -> int | None:
        with self.session() as session:
            value = session.scalar(
                select(LocalMoranRunModel.run_id)
                .order_by(LocalMoranRunModel.run_id.desc())
                .limit(1)
            )
        return None if value is None else int(value)

    def get_run(self, run_id: int) -> LocalMoranRunModel | None:
        with self.session() as session:
            return session.get(LocalMoranRunModel, int(run_id))

    def read_export_rows(self, run_id: int) -> list[dict]:
        sql = text(
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
                o.snap_ok,
                o.is_closed,
                r.metric,
                r.value,
                r.n_window,
                r.n_neighbors,
                r.window_mean,
                r.window_sd,
                r.z_value,
                r.spatial_lag,
                r.local_i,
                r.permutation_expected_i,
                r.permutation_z,
                r.p_sim,
                r.q_fdr_bh,
                r.quadrant,
                r.cluster,
                r.cluster_fdr,
                r.status
            FROM local_moran_results AS r
            INNER JOIN local_moran_point_order AS o
                ON o.run_id = r.run_id
               AND o.point_id = r.point_id
            INNER JOIN points AS p
                ON p.point_id = r.point_id
            WHERE r.run_id = :run_id
            ORDER BY
                o.segment_id,
                o.order_in_segment,
                p.point_id,
                r.metric
            """
        )

        with self.session() as session:
            rows = session.execute(sql, {"run_id": int(run_id)}).mappings().all()

        return [dict(row) for row in rows]
