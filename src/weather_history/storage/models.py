"""
src/weather_history/storage/models.py

ORM-модели для хранения данных о ветре (Open-Meteo).

Компромиссная схема без избыточного копирования:
- Скорость (максимум и среднее), порывы и направление ветра хранятся
  ВМЕСТЕ в одной строке WeatherDayModel (один узел сетки + одна дата =
  одна строка).
- Источник данных (модель реанализа, например "era5") и единицы
  измерения (ws_unit, wd_unit) вынесены в отдельный справочник
  WeatherSourceModel — иначе одни и те же строки "era5", "km/h", "°"
  повторялись бы в каждой строке weather_days (тысячи раз для
  тысяч дат), что является чистым дублированием. Теперь в
  WeatherDayModel есть только внешний ключ source_id.
- DownloadSegmentModel хранит только метаданные скачанного диапазона,
  без копии самих значений ветра (они уже лежат в WeatherDayModel).

Порывы ветра (wind_gusts_10m_max) и средняя скорость (wind_speed_10m_mean)
теперь входят в схему как отдельные колонки WeatherDayModel.

JSON-тип из SQLAlchemy работает одинаково на SQLite и PostgreSQL, поэтому
схема не требует изменений при переходе с sqlite:/// на postgresql+psycopg://.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class WeatherSourceModel(Base):
    """
    Справочник источников данных о ветре: модель реанализа/прогноза
    (например "era5", "gfs") + единицы измерения скорости и направления.

    Вынесено из WeatherDayModel, чтобы не повторять одинаковые строки
    "era5" / "km/h" / "°" в каждой суточной записи — единицы измерения
    и источник модели практически всегда фиксированы для всего набора
    данных, поэтому хранение их построчно было бы избыточным копированием.

    Порывы измеряются в тех же единицах, что и скорость ветра (ws_unit),
    отдельная колонка для единиц порывов не нужна.
    """

    __tablename__ = "weather_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    model: Mapped[str] = mapped_column(String(32), nullable=False)
    ws_unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    wd_unit: Mapped[str | None] = mapped_column(String(16), nullable=True)

    days: Mapped[list["WeatherDayModel"]] = relationship(back_populates="source")
    download_segments: Mapped[list["DownloadSegmentModel"]] = relationship(back_populates="source")

    __table_args__ = (
        UniqueConstraint("model", "ws_unit", "wd_unit", name="uq_weather_source"),
    )


class GridPointModel(Base):
    """Узел регулярной сетки Open-Meteo (0.25° по умолчанию)."""

    __tablename__ = "weather_grid_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Запрошенные координаты (то, что отправили в Open-Meteo)
    req_lat: Mapped[float] = mapped_column(Float, nullable=False)
    req_lon: Mapped[float] = mapped_column(Float, nullable=False)

    # Положение узла относительно основной сетки (буферные кольца)
    ring_y: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ring_x: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Идентификатор ячейки из исходного файла точек (например "37.25_44.75")
    cell_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Координаты, фактически возвращённые Open-Meteo (могут немного отличаться)
    resolved_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolved_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    days: Mapped[list["WeatherDayModel"]] = relationship(
        back_populates="grid_point",
        cascade="all, delete-orphan",
    )
    download_segments: Mapped[list["DownloadSegmentModel"]] = relationship(
        back_populates="grid_point",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("req_lat", "req_lon", "ring_y", "ring_x", name="uq_grid_point"),
        Index("ix_grid_point_coords", "req_lat", "req_lon"),
    )


class WeatherDayModel(Base):
    """
    Одна суточная запись ветра для одного узла сетки: максимальная и средняя
    скорость, максимальный порыв и направление хранятся в одной строке.
    Источник данных и единицы измерения не дублируются построчно — только
    ссылка source_id на WeatherSourceModel.
    """

    __tablename__ = "weather_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    grid_point_id: Mapped[int] = mapped_column(
        ForeignKey("weather_grid_points.id", ondelete="CASCADE"),
        nullable=False,
    )
    grid_point: Mapped[GridPointModel] = relationship(back_populates="days")

    source_id: Mapped[int] = mapped_column(
        ForeignKey("weather_sources.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source: Mapped[WeatherSourceModel] = relationship(back_populates="days")

    obs_date: Mapped[date] = mapped_column(Date, nullable=False)

    wind_speed_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_gust_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("grid_point_id", "source_id", "obs_date", name="uq_weather_day"),
        Index("ix_weather_day_lookup", "grid_point_id", "source_id", "obs_date"),
    )


class DownloadSegmentModel(Base):
    """
    Метаданные о том, какой диапазон дат для какого узла уже был скачан.
    Используется только для проверки "что уже скачано" — без хранения
    копии самих значений ветра (они уже лежат в WeatherDayModel).
    """

    __tablename__ = "weather_download_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    grid_point_id: Mapped[int] = mapped_column(
        ForeignKey("weather_grid_points.id", ondelete="CASCADE"),
        nullable=False,
    )
    grid_point: Mapped[GridPointModel] = relationship(back_populates="download_segments")

    source_id: Mapped[int] = mapped_column(
        ForeignKey("weather_sources.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source: Mapped[WeatherSourceModel] = relationship(back_populates="download_segments")

    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    variables_key: Mapped[str] = mapped_column(String(16), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    cell_selection: Mapped[str] = mapped_column(String(32), nullable=False)

    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    downloaded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "grid_point_id", "source_id", "start_date", "end_date",
            "variables_key", "timezone", "cell_selection",
            name="uq_download_segment",
        ),
        Index(
            "ix_download_segment_lookup",
            "grid_point_id", "source_id", "variables_key", "timezone", "cell_selection",
        ),
    )
