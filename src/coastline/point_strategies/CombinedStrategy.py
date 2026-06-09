from __future__ import annotations

import geopandas as gpd

from src.coastline.point_strategies.PointExtractionStrategy import (
    PointExtractionStrategy,
    PointSource,
)


class CombinedStrategy(PointExtractionStrategy):
    """
    Объединяет результаты нескольких стратегий в один набор точек.

    drop_duplicates=True удаляет геометрически совпадающие точки
    с точностью до `precision` знаков.
    """

    def __init__(
        self,
        strategies: list[PointExtractionStrategy],
        drop_duplicates: bool = True,
        precision: int = 8,
    ) -> None:
        # source не используется напрямую — каждая дочерняя стратегия
        # управляет своим источником самостоятельно
        super().__init__(source=PointSource.ALL_LINES)
        if not strategies:
            raise ValueError("strategies list must not be empty")
        self.strategies = strategies
        self.drop_duplicates = drop_duplicates
        self.precision = precision

    @property
    def name(self) -> str:
        children = ", ".join(s.name for s in self.strategies)
        return f"CombinedStrategy[{children}]"

    def _extract_from_lines(self, gdf: gpd.GeoDataFrame) -> list[dict]:
        # Не используется напрямую; extract() переопределён ниже.
        raise NotImplementedError

    def extract(self, dataset: "CoastlineDataset") -> list[dict]:
        all_records: list[dict] = []

        for strategy in self.strategies:
            records = strategy.extract(dataset)
            for rec in records:
                rec.setdefault("source_strategy", strategy.name)
            all_records.extend(records)

        if not self.drop_duplicates:
            return all_records

        seen: set[tuple[float, float]] = set()
        unique: list[dict] = []
        for rec in all_records:
            geom = rec.get("geometry")
            if geom is None:
                continue
            key = (round(geom.x, self.precision), round(geom.y, self.precision))
            if key in seen:
                continue
            seen.add(key)
            unique.append(rec)

        return unique
