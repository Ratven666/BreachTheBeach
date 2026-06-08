from __future__ import annotations

import math

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

from src.coastline.domain import CoastlineDataset
from src.coastline.domain.CoastlinePointSet import CoastlinePointSet
from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalsSummary


from dataclasses import dataclass


@dataclass(frozen=True)
class CoastlineNormalConfig:
    sea_side: str = "right"          # "left" | "right"
    normal_length_m: float = 200.0   # длина визуализации нормали
    tangent_delta_m: float = 5.0     # шаг для оценки касательной
    working_crs: str | None = None   # если None -> dataset.metric_crs


class CoastlineNormalService:
    """
    Сервис вычисления нормалей к главной береговой линии.

    Идея:
    - все вычисления выполняются в метрической CRS;
    - нормаль считается относительно главной линии dataset.main_gdf;
    - направление нормали задаётся через sea_side:
        * "left"  -> море слева относительно направления линии
        * "right" -> море справа относительно направления линии
    - основной результат возвращается как CoastlineNormalPointSet.
    """

    def __init__(self, config: CoastlineNormalConfig | None = None) -> None:
        self.config = config or CoastlineNormalConfig()
        self._log = logger.bind(cls="CoastlineNormalService")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_points_with_normals(
        self,
        point_set: CoastlinePointSet,
        dataset: CoastlineDataset,
        name: str | None = None,
    ) -> CoastlineNormalPointSet:
        coastline_line, work_crs = self._prepare_main_line(dataset)
        points = self._prepare_points(point_set, work_crs=work_crs)

        records: list[dict] = []

        for idx, row in points.iterrows():
            pt: Point = row.geometry
            info = self._compute_normal_at_point(
                point=pt,
                coastline=coastline_line,
            )

            attrs = row.drop(labels=["geometry"]).to_dict()

            records.append(
                {
                    **attrs,
                    "point_id": int(idx),
                    "chainage_m": info["chainage_m"],
                    "tx": info["tx"],
                    "ty": info["ty"],
                    "nx": info["nx"],
                    "ny": info["ny"],
                    "normal_azimuth_deg": info["normal_azimuth_deg"],
                    "sea_side": self.config.sea_side,
                    "geometry": pt,
                }
            )

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=work_crs)
        result = CoastlineNormalPointSet.from_gdf(
            gdf,
            name=name or f"{point_set.name}_normals",
        )
        self._log.info(f"Built normals for {len(result)} point(s)")
        return result

    def build_normal_lines(
        self,
        point_set: CoastlinePointSet,
        dataset: CoastlineDataset,
        normal_length_m: float | None = None,
    ) -> gpd.GeoDataFrame:
        normals = self.build_points_with_normals(
            point_set=point_set,
            dataset=dataset,
        )

        length = (
            float(normal_length_m)
            if normal_length_m is not None
            else float(self.config.normal_length_m)
        )

        gdf = normals.to_normal_lines_gdf(normal_length_m=length)
        self._log.info(f"Built {len(gdf)} normal line(s)")
        return gdf

    def plot(
        self,
        point_set: CoastlinePointSet,
        dataset: CoastlineDataset,
        figsize: tuple[float, float] = (12, 12),
        coastline_color: str = "black",
        other_color: str = "lightgray",
        point_color: str = "red",
        normal_color: str = "blue",
        linewidth: float = 1.0,
        normal_linewidth: float = 0.8,
    ):
        import matplotlib.pyplot as plt

        coastline_line, work_crs = self._prepare_main_line(dataset)
        main_gdf = dataset.main_gdf.to_crs(work_crs)
        other_gdf = dataset.other_gdf.to_crs(work_crs)
        points = self._prepare_points(point_set, work_crs=work_crs)
        normal_lines = self.build_normal_lines(point_set=point_set, dataset=dataset)

        fig, ax = plt.subplots(figsize=figsize)

        if not other_gdf.empty:
            other_gdf.plot(ax=ax, color=other_color, linewidth=linewidth)

        if not main_gdf.empty:
            main_gdf.plot(ax=ax, color=coastline_color, linewidth=linewidth * 1.5)

        if not normal_lines.empty:
            normal_lines.plot(ax=ax, color=normal_color, linewidth=normal_linewidth)

        if not points.empty:
            points.plot(ax=ax, color=point_color, markersize=12)

        ax.set_title(
            f"Normals to coastline points ({self.config.sea_side}-sea side)"
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        return fig, ax

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_main_line(
        self,
        dataset: CoastlineDataset,
    ) -> tuple[LineString, str]:
        work_crs = self.config.working_crs or str(dataset.metric_crs)

        main = dataset.main_gdf.to_crs(work_crs)
        if main.empty:
            raise ValueError("dataset.main_gdf is empty")

        geoms = [g for g in main.geometry if g is not None and not g.is_empty]
        if not geoms:
            raise ValueError("Main coastline has no valid geometries")

        self._log.info(f"Preparing main coastline from {len(geoms)} geometry object(s)")
        merged = self._merge_to_single_line(geoms)
        self._log.info(
            f"Prepared main coastline as LineString, length={merged.length:.2f} m"
        )
        return merged, work_crs

    def _prepare_points(
        self,
        point_set: CoastlinePointSet,
        work_crs: str,
    ) -> gpd.GeoDataFrame:
        if point_set.gdf.empty:
            raise ValueError("CoastlinePointSet is empty")

        if point_set.gdf.crs is None:
            raise ValueError("CoastlinePointSet.gdf has no CRS")

        return point_set.gdf.to_crs(work_crs).copy()

    def _merge_to_single_line(self, geoms: list) -> LineString:
        parts: list[LineString] = []

        for geom in geoms:
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, LineString):
                if len(geom.coords) >= 2:
                    parts.append(geom)

            elif isinstance(geom, MultiLineString):
                for part in geom.geoms:
                    if part is not None and not part.is_empty and len(part.coords) >= 2:
                        parts.append(part)

            else:
                self._log.warning(f"Skipping unsupported geometry type: {geom.geom_type}")

        if not parts:
            raise ValueError("Main coastline has no valid linear geometries")

        if len(parts) == 1:
            return parts[0]

        multiline = MultiLineString(parts)
        merged = linemerge(multiline)

        if isinstance(merged, LineString):
            return merged

        if isinstance(merged, MultiLineString):
            valid_parts = [g for g in merged.geoms if g is not None and not g.is_empty]
            if not valid_parts:
                raise ValueError("Failed to merge coastline into valid lines")

            longest = max(valid_parts, key=lambda g: g.length)
            self._log.warning(
                f"Main coastline merged to MultiLineString ({len(valid_parts)} parts); "
                f"using longest part, length={longest.length:.2f} m"
            )
            return longest

        raise TypeError(f"Unsupported merged coastline geometry: {merged.geom_type}")

    def _compute_normal_at_point(
        self,
        point: Point,
        coastline: LineString,
    ) -> dict[str, float]:
        s = float(coastline.project(point))
        L = float(coastline.length)

        delta = max(float(self.config.tangent_delta_m), 0.01)

        s0 = max(0.0, s - delta)
        s1 = min(L, s + delta)

        if math.isclose(s0, s1):
            if s <= 0.0:
                s1 = min(L, s + delta)
            else:
                s0 = max(0.0, s - delta)

        p0 = coastline.interpolate(s0)
        p1 = coastline.interpolate(s1)

        dx = p1.x - p0.x
        dy = p1.y - p0.y

        norm_t = math.hypot(dx, dy)
        if norm_t == 0:
            raise ValueError("Failed to compute tangent vector: zero length segment")

        tx = dx / norm_t
        ty = dy / norm_t

        sea_side = self.config.sea_side.lower()
        if sea_side == "left":
            nx = -ty
            ny = tx
        elif sea_side == "right":
            nx = ty
            ny = -tx
        else:
            raise ValueError("sea_side must be 'left' or 'right'")

        norm_n = math.hypot(nx, ny)
        if norm_n == 0:
            raise ValueError("Failed to compute normal vector")

        nx /= norm_n
        ny /= norm_n

        azimuth_deg = (math.degrees(math.atan2(nx, ny)) + 360.0) % 360.0

        return {
            "chainage_m": s,
            "tx": tx,
            "ty": ty,
            "nx": nx,
            "ny": ny,
            "normal_azimuth_deg": azimuth_deg,
        }

if __name__ == "__main__":
    from pathlib import Path

    from src.coastline.point_strategies import EqualStepAlongLineStrategy, PointSource
    from src.coastline.services import CoastlinePointExtractor

    dataset = CoastlineDataset.from_geojson(
        main_path="../../../data/main_coastline.geojson",
        other_path="../../../data/other_lines.geojson",
        name="novoross_coastline",
    )

    extractor = CoastlinePointExtractor()

    points = extractor.extract(
        dataset=dataset,
        strategy=EqualStepAlongLineStrategy(
            step_m=150.0,
            source=PointSource.MAIN_ONLY,
            include_endpoints=True,
            working_crs=str(dataset.metric_crs),
        ),
        name="novoross_main_step_points",
    )

    normal_service = CoastlineNormalService(
        CoastlineNormalConfig(
            sea_side="right",
            normal_length_m=300.0,
            tangent_delta_m=10.0,
            working_crs=str(dataset.metric_crs),
        )
    )

    normal_points = normal_service.build_points_with_normals(
        point_set=points,
        dataset=dataset,
        name="novoross_normals",
    )

    normal_lines = normal_points.to_normal_lines_gdf(normal_length_m=300.0)

    out_dir = Path("../../../output")
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_points.to_geojson(out_dir / "points_with_normals.geojson")
    normal_lines.to_file(out_dir / "normal_lines.geojson", driver="GeoJSON")

    print(normal_points)
    print(normal_points.summary())