from __future__ import annotations

from loguru import logger
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
from src.coastline.services.CoastlineNormalService import CoastlineNormalConfig
from src.coastline.storage.models import (
    COORD_SCALE,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlineSourceModel,
)


class CoastlineNormalRepository:
    """
    Сохраняет и читает CoastlineNormalPointSet из SQLite через SQLAlchemy.

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
        """
        self._assert_point_source_exists(point_source_id)

        crs_str = str(normal_set.gdf.crs) if normal_set.gdf.crs is not None else None

        # 1. Метаданные запуска нормалей
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

        # 2. Точки нормалей — core INSERT (обход bulk_save_objects / BigInteger PK)
        rows_data = [
            {
                "normal_source_id": normal_source.id,
                "seq": int(seq),
                "lon_i": round(float(row.geometry.x) * COORD_SCALE),
                "lat_i": round(float(row.geometry.y) * COORD_SCALE),
                "chainage_m": float(row["chainage_m"]),
                "nx": float(row["nx"]),
                "ny": float(row["ny"]),
                "normal_azimuth_deg": float(row["normal_azimuth_deg"]),
            }
            for seq, row in normal_set.gdf.iterrows()
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
                "id":               r.id,
                "name":             r.name,
                "point_source_id":  r.point_source_id,
                "sea_side":         r.sea_side,
                "tangent_delta_m":  r.tangent_delta_m,
                "working_crs":      r.working_crs,
                "result_crs":       r.result_crs,
                "normals_count":    r.normals_count,
                "created_at":       str(r.created_at),
            }
            for r in rows
        ]

    def load(self, normal_source_id: int) -> CoastlineNormalPointSet:
        """Читает нормали по id CoastlineNormalSourceModel."""
        import geopandas as gpd
        from shapely.geometry import Point

        ns = self._session.get(CoastlineNormalSourceModel, normal_source_id)
        if ns is None:
            raise ValueError(
                f"CoastlineNormalSource id={normal_source_id} not found"
            )

        rows = (
            self._session.execute(
                select(CoastlineNormalModel)
                .where(CoastlineNormalModel.normal_source_id == normal_source_id)
                .order_by(CoastlineNormalModel.seq)
            )
            .scalars()
            .all()
        )

        records = [
            {
                "point_id":           r.point_id,
                "chainage_m":         r.chainage_m,
                "tx":                 r.tx,
                "ty":                 r.ty,
                "nx":                 r.nx,
                "ny":                 r.ny,
                "normal_azimuth_deg": r.normal_azimuth_deg,
                "sea_side":           r.sea_side,
                "geometry":           Point(
                    r.lon_i / COORD_SCALE,
                    r.lat_i / COORD_SCALE,
                ),
            }
            for r in rows
        ]

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=ns.result_crs)
        return CoastlineNormalPointSet.from_gdf(gdf, name=ns.name)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _assert_point_source_exists(self, point_source_id: int) -> None:
        if self._session.get(CoastlineSourceModel, point_source_id) is None:
            raise ValueError(
                f"CoastlineSource id={point_source_id} not found. "
                "Run point import pipeline first."
            )