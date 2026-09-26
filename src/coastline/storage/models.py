from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CoastlineSourceModel(Base):
    __tablename__ = "coastline_sources"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    strategy_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    source_mode: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    strategy_params: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Пути к исходным GeoJSON — используются скриптами
    # расчёта нормалей и длин разгона ветра.
    main_geojson_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )
    other_geojson_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    crs: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    points_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    points: Mapped[list[CoastlinePointModel]] = relationship(
        "CoastlinePointModel",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    normal_sources: Mapped[list[CoastlineNormalSourceModel]] = relationship(
        "CoastlineNormalSourceModel",
        back_populates="source",
        cascade="all, delete-orphan",
    )


class CoastlinePointModel(Base):
    __tablename__ = "coastline_points"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "coastline_sources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    lon: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    lat: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    source: Mapped[CoastlineSourceModel] = relationship(
        "CoastlineSourceModel",
        back_populates="points",
    )
    normals: Mapped[list[CoastlineNormalModel]] = relationship(
        "CoastlineNormalModel",
        back_populates="point",
        cascade="all, delete-orphan",
    )
    wind_fetches: Mapped[list[WindFetchModel]] = relationship(
        "WindFetchModel",
        back_populates="point",
        cascade="all, delete-orphan",
    )


class CoastlineNormalSourceModel(Base):
    __tablename__ = "coastline_normal_sources"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    point_source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "coastline_sources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    sea_side: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    tangent_delta_m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    working_crs: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    result_crs: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    normals_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    normals: Mapped[list[CoastlineNormalModel]] = relationship(
        "CoastlineNormalModel",
        back_populates="normal_source",
        cascade="all, delete-orphan",
    )
    source: Mapped[CoastlineSourceModel] = relationship(
        "CoastlineSourceModel",
        back_populates="normal_sources",
        foreign_keys=[point_source_id],
    )


class CoastlineNormalModel(Base):
    __tablename__ = "coastline_normals"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    normal_source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "coastline_normal_sources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    point_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "coastline_points.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    nx: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    ny: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    normal_azimuth_deg: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    normal_source: Mapped[CoastlineNormalSourceModel] = relationship(
        "CoastlineNormalSourceModel",
        back_populates="normals",
    )
    point: Mapped[CoastlinePointModel] = relationship(
        "CoastlinePointModel",
        back_populates="normals",
    )


class WindFetchModel(Base):
    """
    Рассчитанная длина разгона ветра для одной точки
    и одного абсолютного азимута.

    Одна строка соответствует уникальной паре:
        point_id + azimuth_deg
    """

    __tablename__ = "wind_fetches"

    __table_args__ = (
        UniqueConstraint(
            "point_id",
            "azimuth_deg",
            name="uq_wind_fetches_point_azimuth",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    point_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "coastline_points.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    azimuth_deg: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    fetch_length_m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    point: Mapped[CoastlinePointModel] = relationship(
        "CoastlinePointModel",
        back_populates="wind_fetches",
    )
