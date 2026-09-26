# src/coastline/storage/models.py
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

# Масштаб координат: градусы × COORD_SCALE → целое.
# 1 000 000 → точность ~0.1 м на экваторе.
COORD_SCALE: int = 1_000_000


class CoastlineSourceModel(Base):
    """Метаданные одного экспорта точек береговой линии."""

    __tablename__ = "coastline_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    geojson_path: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    source_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_params: Mapped[str] = mapped_column(Text, nullable=False)
    crs: Mapped[str | None] = mapped_column(String(64), nullable=True)
    points_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    points: Mapped[list["CoastlinePointModel"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "name", "strategy_name", "source_mode",
            name="uq_coastline_source",
        ),
    )


class CoastlinePointModel(Base):
    """
    Одна точка береговой линии.

    lon_i, lat_i — координаты как целые: градусы × COORD_SCALE.
    seq          — индекс точки вдоль линии (0-based); вместе с
                   source_id образует составной первичный ключ.
                   WITHOUT ROWID → кластерный индекс по (source_id, seq).
    """

    __tablename__ = "coastline_points"
    __table_args__ = {"sqlite_with_rowid": False}

    source_id: Mapped[int] = mapped_column(
        ForeignKey("coastline_sources.id", ondelete="CASCADE"),
        primary_key=True,
    )
    source: Mapped[CoastlineSourceModel] = relationship(back_populates="points")

    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    lon_i: Mapped[int] = mapped_column(Integer, nullable=False)
    lat_i: Mapped[int] = mapped_column(Integer, nullable=False)
