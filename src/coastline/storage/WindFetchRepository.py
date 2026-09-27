from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from src.coastline.storage.models import (
    CoastlinePointModel,
    WindFetchModel,
)
from src.wind_fetch.models import MultiDirectionFetchResult


class WindFetchRepository:
    """
    Репозиторий рассчитанных длин разгона ветра.

    В таблице wind_fetches хранится одна строка
    для каждой пары point_id + azimuth_deg.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._log = logger.bind(cls=self.__class__.__name__)

    def replace_for_points(
        self,
        results: Sequence[MultiDirectionFetchResult],
    ) -> int:
        """
        Заменяет результаты для всех точек, присутствующих в results.

        Перед записью:
        - проверяется существование точек;
        - проверяется отсутствие повторяющихся пар
          point_id + azimuth_deg;
        - удаляются ранее рассчитанные направления этих точек.

        Метод не выполняет commit. Управление транзакцией остаётся
        у вызывающего кода.
        """
        if not results:
            return 0

        point_ids = {
            int(result.point_id)
            for result in results
        }

        existing_point_ids = set(
            self._session.execute(
                select(CoastlinePointModel.id).where(
                    CoastlinePointModel.id.in_(point_ids)
                )
            ).scalars()
        )

        missing_point_ids = sorted(
            point_ids - existing_point_ids
        )

        if missing_point_ids:
            raise ValueError(
                "Coastline points not found: "
                f"{missing_point_ids[:10]}"
            )

        rows: list[dict[str, int | float]] = []
        unique_keys: set[tuple[int, float]] = set()

        for result in results:
            point_id = int(result.point_id)
            azimuth_deg = float(result.azimuth_deg) % 360.0

            unique_key = (
                point_id,
                azimuth_deg,
            )

            if unique_key in unique_keys:
                raise ValueError(
                    "Duplicate wind-fetch result for "
                    f"point_id={point_id}, "
                    f"azimuth_deg={azimuth_deg}"
                )

            unique_keys.add(unique_key)

            rows.append(
                {
                    "point_id": point_id,
                    "azimuth_deg": azimuth_deg,
                    "fetch_length_m": round(
                        result.fetch_length_m
                    ),
                }
            )

        self._session.execute(
            delete(WindFetchModel).where(
                WindFetchModel.point_id.in_(point_ids)
            )
        )

        self._session.execute(
            insert(WindFetchModel),
            rows,
        )

        self._log.info(
            f"Prepared {len(rows)} wind-fetch rows "
            f"for {len(point_ids)} points"
        )

        return len(rows)

    def load_for_point(
        self,
        point_id: int,
    ) -> list[WindFetchModel]:
        """
        Возвращает рассчитанные направления одной точки,
        отсортированные по возрастанию азимута.
        """
        rows = (
            self._session.execute(
                select(WindFetchModel)
                .where(
                    WindFetchModel.point_id == point_id
                )
                .order_by(
                    WindFetchModel.azimuth_deg
                )
            )
            .scalars()
            .all()
        )

        return list(rows)
