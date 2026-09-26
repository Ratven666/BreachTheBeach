from __future__ import annotations

import datetime

from sqlalchemy import (
    BigInteger, DateTime, ForeignKey, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

COORD_SCALE: int = 10_000_000  # точность ~1 см на экваторе


class Base(DeclarativeBase):
    pass


class CoastlineSourceModel(Base):
    __tablename__ = "coastline_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    geojson_path: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_params: Mapped[str | None] = mapped_column(Text, nullable=True)
    crs: Mapped[str | None] = mapped_column(String(64), nullable=True)
    points_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    points: Mapped[list[CoastlinePointModel]] = relationship(
        "CoastlinePointModel",
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CoastlinePointModel(Base):
    __tablename__ = "coastline_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("coastline_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    lon_i: Mapped[int] = mapped_column(BigInteger, nullable=False)
    lat_i: Mapped[int] = mapped_column(BigInteger, nullable=False)

    source: Mapped[CoastlineSourceModel] = relationship(
        "CoastlineSourceModel", back_populates="points"
    )
