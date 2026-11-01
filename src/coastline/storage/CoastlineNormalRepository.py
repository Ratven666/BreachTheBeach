from __future__ import annotations

import geopandas as gpd
from loguru import logger
from shapely.geometry import Point
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
from src.coastline.services.CoastlineNormalService import CoastlineNormalConfig
from src.coastline.storage.models import (
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
)


class CoastlineNormalRepository:
    """
    Сохраняет и читает нормали береговой линии через SQLAlchemy.

    В БД хранятся только поля CoastlineNormalModel:
        normal_source_id, point_id, nx, ny, normal_azimuth_deg.
    Координаты точек (lon/lat) берутся через JOIN с coastline_points.
    Поля chainage_m, tx, ty, sea_side в БД не хранятся.

    Пример
    ------
    with SessionLocal() as session:
        repo = CoastlineNormalRepository(session)
        normal_source_id = repo.save(
            normal_set=normal_points,
            point_source_id=1,
            config=config,
            name="nvrsk_normals_step200m_right",
        )
        gdf = repo.load_as_gdf(normal_source_id)
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._log = logger.bind(cls=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        normal_set: CoastlineNormalPointSet,
        point_source_id: int,
        config: CoastlineNormalConfig,
        name: str,
    ) -> int:
        """
        Сохраняет CoastlineNormalPointSet в БД.
        Возвращает id созданной записи CoastlineNormalSourceModel.

        Ожидает, что в normal_set.gdf есть колонка point_id —
        FK на coastline_points.id (заполняется CoastlineNormalService).
        """
        self._assert_point_source_exists(point_source_id)

        crs_str = str(normal_set.gdf.crs) if normal_set.gdf.crs is not None else None

        # 1. Метаданные набора нормалей
        normal_source = CoastlineNormalSourceModel(
            name=name,
            point_source_id=point_source_id,
            sea_side=config.sea_side,
            tangent_delta_m=float(config.tangent_delta_m),
            working_crs=str(config.working_crs) if config.working_crs else None,
            result_crs=crs_str,
            normals_count=len(normal_set.gdf),
        )
        self._session.add(normal_source)
        self._session.flush()  # получаем normal_source.id

        # 2. Строки нормалей — только поля CoastlineNormalModel
        rows_data = [
            {
                "normal_source_id":   normal_source.id,
                "point_id":           int(row["point_id"]),
                "nx":                 float(row["nx"]),
                "ny":                 float(row["ny"]),
                "normal_azimuth_deg": float(row["normal_azimuth_deg"]),
            }
            for _, row in normal_set.gdf.iterrows()
        ]
        self._session.execute(insert(CoastlineNormalModel), rows_data)
        self._session.commit()

        self._log.success(
            f"Saved {len(rows_data)} normals → "
            f"normal_source id={normal_source.id} name={name!r}"
        )
        return normal_source.id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_normal_sources(self) -> list[dict]:
        """Список всех записей CoastlineNormalSourceModel."""
        rows = (
            self._session.execute(
                select(CoastlineNormalSourceModel).order_by(
                    CoastlineNormalSourceModel.id
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "id":              r.id,
                "name":            r.name,
                "point_source_id": r.point_source_id,
                "sea_side":        r.sea_side,
                "tangent_delta_m": r.tangent_delta_m,
                "working_crs":     r.working_crs,
                "result_crs":      r.result_crs,
                "normals_count":   r.normals_count,
                "created_at":      str(r.created_at),
            }
            for r in rows
        ]

    def load_as_gdf(self, normal_source_id: int) -> gpd.GeoDataFrame:
        """
        Читает нормали по id и возвращает GeoDataFrame в EPSG:4326.

        Колонки: normal_id, normal_source_id, point_id,
                 nx, ny, normal_azimuth_deg, geometry (Point, EPSG:4326).

        Координаты точек (p.lon / p.lat) хранятся в coastline_points
        в WGS-84 градусах, поэтому CRS результата всегда EPSG:4326,
        независимо от result_crs набора нормалей (та хранит метрическую
        CRS вычислений и не связана с форматом хранения координат).
        """
        ns = self._session.get(CoastlineNormalSourceModel, normal_source_id)
        if ns is None:
            raise ValueError(
                f"CoastlineNormalSource id={normal_source_id} not found"
            )

        rows = (
            self._session.execute(
                select(CoastlineNormalModel, CoastlinePointModel)
                .join(
                    CoastlinePointModel,
                    CoastlineNormalModel.point_id == CoastlinePointModel.id,
                )
                .where(CoastlineNormalModel.normal_source_id == normal_source_id)
                .order_by(CoastlineNormalModel.id)
            )
            .all()
        )

        records = [
            {
                "normal_id":          n.id,
                "normal_source_id":   n.normal_source_id,
                "point_id":           n.point_id,   # DB-id → FK в wind_fetches
                "nx":                 n.nx,
                "ny":                 n.ny,
                "normal_azimuth_deg": n.normal_azimuth_deg,
                # p.lon / p.lat — градусы WGS-84; CRS явно EPSG:4326
                "geometry":           Point(p.lon, p.lat),
            }
            for n, p in rows
        ]

        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _assert_point_source_exists(self, point_source_id: int) -> None:
        if self._session.get(CoastlineSourceModel, point_source_id) is None:
            raise ValueError(
                f"CoastlineSource id={point_source_id} not found. "
                "Run point import pipeline first."
            )
