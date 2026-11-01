from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, Index, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LocalMoranRunModel(Base):
    __tablename__ = "local_moran_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_utc: Mapped[str] = mapped_column(String, nullable=False)

    coastline_path: Mapped[str] = mapped_column(String, nullable=False)
    coastline_sha256: Mapped[str] = mapped_column(String, nullable=False)
    coastline_crs: Mapped[str] = mapped_column(String, nullable=False)
    metric_crs: Mapped[str] = mapped_column(String, nullable=False)

    radius_m: Mapped[float] = mapped_column(Float, nullable=False)
    window_width_m: Mapped[float] = mapped_column(Float, nullable=False)
    max_snap_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    weight_mode: Mapped[str] = mapped_column(String, nullable=False)
    permutations: Mapped[int] = mapped_column(Integer, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)

    alpha_raw: Mapped[float] = mapped_column(Float, nullable=False)
    alpha_fdr: Mapped[float] = mapped_column(Float, nullable=False)

    alongshore_metric: Mapped[str] = mapped_column(String, nullable=False)
    strict_snap: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    n_points_total: Mapped[int] = mapped_column(Integer, nullable=False)
    n_points_snap_ok: Mapped[int] = mapped_column(Integer, nullable=False)
    n_results: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("radius_m > 0.0"),
        CheckConstraint("window_width_m > 0.0"),
        CheckConstraint("permutations >= 99"),
        CheckConstraint("random_seed >= 0"),
        CheckConstraint("alpha_raw > 0.0 AND alpha_raw < 1.0"),
        CheckConstraint("alpha_fdr > 0.0 AND alpha_fdr < 1.0"),
        CheckConstraint("strict_snap IN (0, 1)"),
        CheckConstraint("n_points_total >= 0"),
        CheckConstraint("n_points_snap_ok >= 0"),
        CheckConstraint("n_results >= 0"),
        CheckConstraint("weight_mode IN ('binary', 'inverse_distance')"),
        CheckConstraint(
            "alongshore_metric IN "
            "('normal_mean_wm', 'alongshore_abs_mean_wm', 'alongshore_net_mean_wm')"
        ),
    )


class LocalMoranPointOrderModel(Base):
    __tablename__ = "local_moran_point_order"

    run_id: Mapped[int] = mapped_column(
        ForeignKey("local_moran_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    point_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    segment_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    order_in_segment: Mapped[int] = mapped_column(Integer, nullable=False)
    chainage_m: Mapped[float] = mapped_column(Float, nullable=False)
    segment_length_m: Mapped[float] = mapped_column(Float, nullable=False)
    snap_distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    snap_ok: Mapped[int] = mapped_column(Integer, nullable=False)
    is_closed: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("order_in_segment >= 0"),
        CheckConstraint("chainage_m >= 0.0"),
        CheckConstraint("segment_length_m >= 0.0"),
        CheckConstraint("snap_distance_m >= 0.0"),
        CheckConstraint("snap_ok IN (0, 1)"),
        CheckConstraint("is_closed IN (0, 1)"),
        Index(
            "idx_local_moran_order_run_segment_order",
            "run_id",
            "segment_id",
            "order_in_segment",
        ),
    )


class LocalMoranResultModel(Base):
    __tablename__ = "local_moran_results"

    run_id: Mapped[int] = mapped_column(
        ForeignKey("local_moran_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    point_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric: Mapped[str] = mapped_column(String, primary_key=True)

    value: Mapped[float | None] = mapped_column(Float, nullable=True)

    n_window: Mapped[int] = mapped_column(Integer, nullable=False)
    n_neighbors: Mapped[int] = mapped_column(Integer, nullable=False)

    window_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_sd: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    spatial_lag: Mapped[float | None] = mapped_column(Float, nullable=True)
    local_i: Mapped[float | None] = mapped_column(Float, nullable=True)

    permutation_expected_i: Mapped[float | None] = mapped_column(Float, nullable=True)
    permutation_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_sim: Mapped[float | None] = mapped_column(Float, nullable=True)
    q_fdr_bh: Mapped[float | None] = mapped_column(Float, nullable=True)

    quadrant: Mapped[str | None] = mapped_column(String, nullable=True)
    cluster: Mapped[str | None] = mapped_column(String, nullable=True)
    cluster_fdr: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        CheckConstraint("n_window >= 0"),
        CheckConstraint("n_neighbors >= 0"),
        CheckConstraint(
            "metric IN "
            "('normal_mean_wm', 'alongshore_abs_mean_wm', 'alongshore_net_mean_wm')"
        ),
        Index("idx_local_moran_results_run_metric", "run_id", "metric"),
        Index("idx_local_moran_results_run_metric_cluster", "run_id", "metric", "cluster_fdr"),
    )
