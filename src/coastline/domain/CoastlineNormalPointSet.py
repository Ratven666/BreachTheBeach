from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class CoastlineNormalsSummary:
    count: int
    crs: str | None
    min_chainage_m: float | None
    max_chainage_m: float | None
    mean_normal_azimuth_deg: float | None
    sea_side: str | None


@dataclass(frozen=True)
class CoastlineNormalsValidationReport:
    count: int
    has_nulls: bool
    duplicated_point_ids: int
    invalid_tangent_count: int
    invalid_normal_count: int
    invalid_azimuth_count: int
    chainage_not_sorted: bool
    min_tangent_norm: float | None
    max_tangent_norm: float | None
    min_normal_norm: float | None
    max_normal_norm: float | None

    @property
    def is_valid(self) -> bool:
        return (
            not self.has_nulls
            and self.duplicated_point_ids == 0
            and self.invalid_tangent_count == 0
            and self.invalid_normal_count == 0
            and self.invalid_azimuth_count == 0
        )


class CoastlineNormalPointSet:
    """
    Domain-класс для набора точек береговой линии с вычисленными нормалями.

    Обязательные поля GeoDataFrame:
    - geometry : Point
    - point_id : int
    - chainage_m : float
    - tx, ty : float
    - nx, ny : float
    - normal_azimuth_deg : float
    - sea_side : str
    """

    REQUIRED_COLUMNS = {
        "point_id",
        "chainage_m",
        "tx",
        "ty",
        "nx",
        "ny",
        "normal_azimuth_deg",
        "sea_side",
        "geometry",
    }

    NUMERIC_COLUMNS = {
        "point_id",
        "chainage_m",
        "tx",
        "ty",
        "nx",
        "ny",
        "normal_azimuth_deg",
    }

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        name: str | None = None,
    ) -> None:
        self._log = logger.bind(cls=self.__class__.__name__)
        self.name = name or "coastline_normal_points"
        self.gdf = self._validate_gdf(gdf.copy())
        self._log.debug(
            f"Initialized: name={self.name}, count={len(self.gdf)}, crs={self.gdf.crs}"
        )

    @classmethod
    def from_gdf(
        cls,
        gdf: gpd.GeoDataFrame,
        name: str | None = None,
    ) -> "CoastlineNormalPointSet":
        return cls(gdf=gdf, name=name)

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        name: str | None = None,
    ) -> "CoastlineNormalPointSet":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normal points file not found: {path}")

        logger.bind(cls=cls.__name__).info(f"Reading normal points: {path}")
        gdf = gpd.read_file(path)
        return cls(gdf=gdf, name=name or path.stem)

    @property
    def crs(self):
        return self.gdf.crs

    @property
    def empty(self) -> bool:
        return self.gdf.empty

    @property
    def count(self) -> int:
        return len(self.gdf)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if self.gdf.empty:
            raise ValueError("Point set is empty")
        return tuple(self.gdf.total_bounds)

    @property
    def sea_side(self) -> str | None:
        if self.gdf.empty or "sea_side" not in self.gdf.columns:
            return None

        values = self.gdf["sea_side"].dropna().unique().tolist()
        if not values:
            return None
        if len(values) == 1:
            return str(values[0])

        return ",".join(map(str, values))

    def _validate_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("gdf must be a GeoDataFrame")

        if gdf.empty:
            raise ValueError("GeoDataFrame is empty")

        if gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS")

        missing = self.REQUIRED_COLUMNS - set(gdf.columns)
        if missing:
            raise ValueError(
                f"Normal point GeoDataFrame is missing required columns: {sorted(missing)}"
            )

        invalid_geom = gdf.geometry.isna().any() or any(
            geom is None or geom.is_empty for geom in gdf.geometry
        )
        if invalid_geom:
            raise ValueError("GeoDataFrame contains empty geometries")

        bad_types = [
            geom.geom_type
            for geom in gdf.geometry
            if not isinstance(geom, Point)
        ]
        if bad_types:
            raise TypeError(
                f"All geometries must be Point. Found: {sorted(set(bad_types))}"
            )

        for col in self.NUMERIC_COLUMNS:
            coerced = pd.to_numeric(gdf[col], errors="coerce")
            if coerced.isna().any():
                raise TypeError(f"Column '{col}' must contain only numeric values")
            gdf[col] = coerced

        gdf["sea_side"] = gdf["sea_side"].astype(str)

        return gdf.reset_index(drop=True)

    def to_crs(self, crs: str) -> "CoastlineNormalPointSet":
        return CoastlineNormalPointSet(
            gdf=self.gdf.to_crs(crs),
            name=self.name,
        )

    def copy(self) -> "CoastlineNormalPointSet":
        return CoastlineNormalPointSet(
            gdf=self.gdf.copy(),
            name=self.name,
        )

    def sort_by_chainage(self, ascending: bool = True) -> "CoastlineNormalPointSet":
        gdf = self.gdf.sort_values("chainage_m", ascending=ascending).reset_index(drop=True)
        return CoastlineNormalPointSet(gdf=gdf, name=self.name)

    def subset_by_chainage(
        self,
        start_m: float | None = None,
        end_m: float | None = None,
    ) -> "CoastlineNormalPointSet":
        gdf = self.gdf.copy()

        if start_m is not None:
            gdf = gdf[gdf["chainage_m"] >= float(start_m)]

        if end_m is not None:
            gdf = gdf[gdf["chainage_m"] <= float(end_m)]

        if gdf.empty:
            raise ValueError("Subset by chainage produced empty result")

        return CoastlineNormalPointSet(gdf=gdf, name=self.name)

    def to_normal_lines_gdf(
        self,
        normal_length_m: float,
    ) -> gpd.GeoDataFrame:
        if normal_length_m <= 0:
            raise ValueError("normal_length_m must be > 0")

        records: list[dict] = []

        for _, row in self.gdf.iterrows():
            p: Point = row.geometry
            nx = float(row["nx"])
            ny = float(row["ny"])

            end = Point(
                p.x + nx * normal_length_m,
                p.y + ny * normal_length_m,
            )

            attrs = row.drop(labels=["geometry"]).to_dict()
            records.append(
                {
                    **attrs,
                    "normal_length_m": float(normal_length_m),
                    "geometry": LineString([p, end]),
                }
            )

        return gpd.GeoDataFrame(records, geometry="geometry", crs=self.gdf.crs)

    def to_geojson(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gdf.to_file(path, driver="GeoJSON")
        self._log.info(f"Saved normal points to {path}")

    def to_gpkg(
        self,
        path: str | Path,
        layer: str = "normal_points",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gdf.to_file(path, layer=layer, driver="GPKG")
        self._log.info(f"Saved normal points to {path} | layer={layer}")

    def export_normal_lines_geojson(
        self,
        path: str | Path,
        normal_length_m: float,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines_gdf = self.to_normal_lines_gdf(normal_length_m=normal_length_m)
        lines_gdf.to_file(path, driver="GeoJSON")
        self._log.info(f"Saved normal lines to {path}")

    def summary(self) -> CoastlineNormalsSummary:
        az = self.gdf["normal_azimuth_deg"].dropna()
        ch = self.gdf["chainage_m"].dropna()

        return CoastlineNormalsSummary(
            count=len(self.gdf),
            crs=str(self.gdf.crs) if self.gdf.crs is not None else None,
            min_chainage_m=float(ch.min()) if not ch.empty else None,
            max_chainage_m=float(ch.max()) if not ch.empty else None,
            mean_normal_azimuth_deg=float(az.mean()) if not az.empty else None,
            sea_side=self.sea_side,
        )

    def validate_vectors(
        self,
        tol: float = 1e-6,
    ) -> CoastlineNormalsValidationReport:
        df = self.gdf.copy()

        tnorm = (df["tx"] ** 2 + df["ty"] ** 2) ** 0.5
        nnorm = (df["nx"] ** 2 + df["ny"] ** 2) ** 0.5

        invalid_tangent = (~tnorm.sub(1.0).abs().le(tol)).sum()
        invalid_normal = (~nnorm.sub(1.0).abs().le(tol)).sum()
        invalid_azimuth = (
            (~df["normal_azimuth_deg"].between(0.0, 360.0, inclusive="left"))
        ).sum()

        chainage_not_sorted = not df["chainage_m"].is_monotonic_increasing
        has_nulls = df[list(self.REQUIRED_COLUMNS)].isna().any().any()
        duplicated_point_ids = int(df["point_id"].duplicated().sum())

        return CoastlineNormalsValidationReport(
            count=len(df),
            has_nulls=bool(has_nulls),
            duplicated_point_ids=duplicated_point_ids,
            invalid_tangent_count=int(invalid_tangent),
            invalid_normal_count=int(invalid_normal),
            invalid_azimuth_count=int(invalid_azimuth),
            chainage_not_sorted=bool(chainage_not_sorted),
            min_tangent_norm=float(tnorm.min()) if len(df) else None,
            max_tangent_norm=float(tnorm.max()) if len(df) else None,
            min_normal_norm=float(nnorm.min()) if len(df) else None,
            max_normal_norm=float(nnorm.max()) if len(df) else None,
        )

    def head_text(self, n: int = 5) -> str:
        cols = [
            "point_id",
            "chainage_m",
            "tx",
            "ty",
            "nx",
            "ny",
            "normal_azimuth_deg",
            "sea_side",
        ]
        use_cols = [c for c in cols if c in self.gdf.columns]
        return self.gdf[use_cols].head(n).to_string(index=False)

    def debug_report(self, n: int = 5) -> str:
        s = self.summary()
        v = self.validate_vectors()

        parts = [
            f"Name: {self.name}",
            f"Count: {s.count}",
            f"CRS: {s.crs}",
            f"Bounds: {self.bounds}",
            f"Sea side: {s.sea_side}",
            f"Chainage range: {s.min_chainage_m} .. {s.max_chainage_m}",
            f"Mean azimuth: {s.mean_normal_azimuth_deg}",
            f"Validation is_valid: {v.is_valid}",
            f"Validation has_nulls: {v.has_nulls}",
            f"Validation duplicated_point_ids: {v.duplicated_point_ids}",
            f"Validation invalid_tangent_count: {v.invalid_tangent_count}",
            f"Validation invalid_normal_count: {v.invalid_normal_count}",
            f"Validation invalid_azimuth_count: {v.invalid_azimuth_count}",
            f"Validation chainage_not_sorted: {v.chainage_not_sorted}",
            f"Tangent norm range: {v.min_tangent_norm} .. {v.max_tangent_norm}",
            f"Normal norm range: {v.min_normal_norm} .. {v.max_normal_norm}",
            "",
            f"Head({n}):",
            self.head_text(n=n),
        ]
        return "\n".join(parts)

    def print_debug(self, n: int = 5) -> None:
        text = self.debug_report(n=n)
        print(text)
        self._log.info("\n" + text)

    def info(self) -> str:
        s = self.summary()
        return (
            f"CoastlineNormalPointSet(name={self.name}, count={s.count}, crs={s.crs}, "
            f"chainage=[{s.min_chainage_m}, {s.max_chainage_m}], sea_side={s.sea_side})"
        )

    def __len__(self) -> int:
        return len(self.gdf)

    def __repr__(self) -> str:
        try:
            return self.info()
        except Exception:
            return f"CoastlineNormalPointSet(name={self.name})"

if __name__ == "__main__":
    from pathlib import Path

    import geopandas as gpd

    # импортируй свой класс так, как он лежит в проекте
    # например:
    # from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet



    def demo_normal_point_set() -> None:
        input_path = Path("../../../output/points_with_normals.geojson")
        output_dir = Path("../../../output/demo")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Чтение из файла
        normal_points = CoastlineNormalPointSet.from_geojson(
            path=input_path,
            name="novoross_normals_demo",
        )

        # 2. Короткая информация об объекте
        print("=== OBJECT ===")
        print(normal_points)
        print()

        # 3. Полный отладочный отчёт
        print("=== DEBUG REPORT ===")
        print(normal_points.debug_report(n=10))
        print()

        # 4. Отдельная проверка валидности
        report = normal_points.validate_vectors()
        print("=== VALIDATION ===")
        print(f"is_valid = {report.is_valid}")
        print(f"count = {report.count}")
        print(f"has_nulls = {report.has_nulls}")
        print(f"duplicated_point_ids = {report.duplicated_point_ids}")
        print(f"invalid_tangent_count = {report.invalid_tangent_count}")
        print(f"invalid_normal_count = {report.invalid_normal_count}")
        print(f"invalid_azimuth_count = {report.invalid_azimuth_count}")
        print(f"chainage_not_sorted = {report.chainage_not_sorted}")
        print(
            f"tangent norm range = {report.min_tangent_norm} .. {report.max_tangent_norm}"
        )
        print(
            f"normal norm range = {report.min_normal_norm} .. {report.max_normal_norm}"
        )
        print()

        # 5. Сортировка по chainage
        normal_points_sorted = normal_points.sort_by_chainage()

        print("=== SORTED HEAD ===")
        print(normal_points_sorted.head_text(n=10))
        print()

        # 6. Подмножество по chainage
        subset = normal_points_sorted.subset_by_chainage(
            start_m=0.0,
            end_m=2000.0,
        )

        print("=== SUBSET ===")
        print(subset)
        print(subset.head_text(n=10))
        print()

        # 7. Построение линий нормалей
        normal_lines_gdf = normal_points_sorted.to_normal_lines_gdf(
            normal_length_m=300.0
        )

        print("=== NORMAL LINES ===")
        print(normal_lines_gdf.head(10).to_string(index=False))
        print()

        # 8. Экспорт результатов
        normal_points_sorted.to_geojson(output_dir / "normal_points_sorted.geojson")
        normal_points_sorted.to_gpkg(
            output_dir / "normal_points_sorted.gpkg",
            layer="normal_points",
        )
        normal_points_sorted.export_normal_lines_geojson(
            output_dir / "normal_lines_300m.geojson",
            normal_length_m=300.0,
        )

        # если хочешь отдельно сохранить subset
        subset.to_geojson(output_dir / "normal_points_subset_0_2000m.geojson")

        print("=== DONE ===")
        print(f"Saved outputs to: {output_dir.resolve()}")


    if __name__ == "__main__":
        demo_normal_point_set()