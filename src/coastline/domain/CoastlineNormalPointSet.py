from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
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
    invalid_orthogonality_count: int
    invalid_side_count: int
    invalid_azimuth_count: int
    invalid_azimuth_consistency_count: int
    chainage_not_sorted: bool          # информационное поле
    min_tangent_norm: float | None
    max_tangent_norm: float | None
    min_normal_norm: float | None
    max_normal_norm: float | None
    max_abs_dot_product: float | None

    @property
    def is_valid(self) -> bool:
        # chainage_not_sorted намеренно не участвует: порядок точек
        # по seq не обязан совпадать с направлением каждой линии.
        return (
            not self.has_nulls
            and self.duplicated_point_ids == 0
            and self.invalid_tangent_count == 0
            and self.invalid_normal_count == 0
            and self.invalid_orthogonality_count == 0
            and self.invalid_side_count == 0
            and self.invalid_azimuth_count == 0
            and self.invalid_azimuth_consistency_count == 0
        )


class CoastlineNormalPointSet:
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

    _GEOJSON_CRS = "EPSG:4326"

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        name: str | None = None,
    ) -> None:
        self._log = logger.bind(cls=self.__class__.__name__)
        self.name = name or "coastline_normal_points"
        self.gdf = self._validate_gdf(gdf.copy())

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
        return cls(gdf=gpd.read_file(path), name=name or path.stem)

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
        return tuple(float(v) for v in self.gdf.total_bounds)

    @property
    def sea_side(self) -> str | None:
        values = self.gdf["sea_side"].dropna().unique().tolist()
        if not values:
            return None
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
                "Normal point GeoDataFrame is missing required columns: "
                f"{sorted(missing)}"
            )

        if gdf.geometry.isna().any():
            raise ValueError("GeoDataFrame contains null geometries")
        if any(g is None or g.is_empty for g in gdf.geometry):
            raise ValueError("GeoDataFrame contains empty geometries")
        if not all(isinstance(g, Point) for g in gdf.geometry):
            raise TypeError("All geometries must be Point")

        for column in self.NUMERIC_COLUMNS:
            values = pd.to_numeric(gdf[column], errors="coerce")
            if values.isna().any():
                raise TypeError(
                    f"Column {column!r} must contain only numeric values"
                )
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Column {column!r} contains NaN or infinity"
                )
            gdf[column] = values

        gdf["point_id"] = gdf["point_id"].astype(int)
        gdf["sea_side"] = gdf["sea_side"].astype(str).str.lower()

        invalid_sides = ~gdf["sea_side"].isin({"left", "right"})
        if invalid_sides.any():
            values = sorted(gdf.loc[invalid_sides, "sea_side"].unique())
            raise ValueError(
                f"Invalid sea_side values: {values}; "
                "expected 'left' or 'right'"
            )

        return gdf.reset_index(drop=True)

    def to_crs(self, crs: str) -> "CoastlineNormalPointSet":
        """
        Перепроецирует геометрию. Компоненты nx/ny остаются в исходной
        working CRS, поэтому линии нормалей строятся до перепроецирования.
        """
        return CoastlineNormalPointSet(
            gdf=self.gdf.to_crs(crs), name=self.name
        )

    def copy(self) -> "CoastlineNormalPointSet":
        return CoastlineNormalPointSet(gdf=self.gdf.copy(), name=self.name)

    def sort_by_chainage(
        self,
        ascending: bool = True,
    ) -> "CoastlineNormalPointSet":
        sort_columns = []
        if "reference_part_id" in self.gdf.columns:
            sort_columns.append("reference_part_id")
        sort_columns.append("chainage_m")

        gdf = self.gdf.sort_values(
            sort_columns, ascending=ascending
        ).reset_index(drop=True)
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

    def to_normal_lines_gdf(self, normal_length_m: float) -> gpd.GeoDataFrame:
        if normal_length_m <= 0:
            raise ValueError("normal_length_m must be > 0")
        if self.gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS")
        if self.gdf.crs.is_geographic:
            raise ValueError(
                "Normal lines cannot be constructed in a geographic CRS; "
                "use the original projected working CRS"
            )

        records: list[dict] = []
        for _, row in self.gdf.iterrows():
            start: Point = row.geometry
            nx = float(row["nx"])
            ny = float(row["ny"])
            end = Point(
                start.x + nx * normal_length_m,
                start.y + ny * normal_length_m,
            )
            attrs = row.drop(labels=["geometry"]).to_dict()
            records.append(
                {
                    **attrs,
                    "normal_length_m": float(normal_length_m),
                    "geometry": LineString([start, end]),
                }
            )

        return gpd.GeoDataFrame(
            records, geometry="geometry", crs=self.gdf.crs
        )

    def to_geojson(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gdf.to_crs(self._GEOJSON_CRS).to_file(path, driver="GeoJSON")

    def to_gpkg(
        self,
        path: str | Path,
        layer: str = "normal_points",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gdf.to_file(path, layer=layer, driver="GPKG")

    def export_normal_lines_geojson(
        self,
        path: str | Path,
        normal_length_m: float,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = self.to_normal_lines_gdf(normal_length_m)
        lines.to_crs(self._GEOJSON_CRS).to_file(path, driver="GeoJSON")

    def summary(self) -> CoastlineNormalsSummary:
        chainage = self.gdf["chainage_m"].dropna()

        mean_nx = float(self.gdf["nx"].mean())
        mean_ny = float(self.gdf["ny"].mean())
        mean_norm = math.hypot(mean_nx, mean_ny)

        mean_azimuth = None
        if mean_norm > 1e-12:
            mean_azimuth = (
                math.degrees(math.atan2(mean_nx, mean_ny)) + 360.0
            ) % 360.0

        return CoastlineNormalsSummary(
            count=len(self.gdf),
            crs=str(self.gdf.crs),
            min_chainage_m=float(chainage.min()) if not chainage.empty else None,
            max_chainage_m=float(chainage.max()) if not chainage.empty else None,
            mean_normal_azimuth_deg=mean_azimuth,
            sea_side=self.sea_side,
        )

    def validate_vectors(
        self,
        tol: float = 1e-6,
        azimuth_tol_deg: float = 1e-6,
    ) -> CoastlineNormalsValidationReport:
        df = self.gdf.copy()

        tx = df["tx"].astype(float)
        ty = df["ty"].astype(float)
        nx = df["nx"].astype(float)
        ny = df["ny"].astype(float)

        tangent_norm = np.hypot(tx, ty)
        normal_norm = np.hypot(nx, ny)
        dot_product = tx * nx + ty * ny

        invalid_tangent = np.abs(tangent_norm - 1.0) > tol
        invalid_normal = np.abs(normal_norm - 1.0) > tol
        invalid_orthogonality = np.abs(dot_product) > tol

        is_left = df["sea_side"].eq("left")
        expected_nx = np.where(is_left, -ty, ty)
        expected_ny = np.where(is_left, tx, -tx)
        side_error = np.hypot(nx - expected_nx, ny - expected_ny)
        invalid_side = side_error > tol

        azimuth = df["normal_azimuth_deg"].astype(float)
        invalid_azimuth = ~azimuth.between(0.0, 360.0, inclusive="left")

        expected_azimuth = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
        azimuth_difference = np.abs(
            (azimuth - expected_azimuth + 180.0) % 360.0 - 180.0
        )
        invalid_azimuth_consistency = azimuth_difference > azimuth_tol_deg

        if "reference_part_id" in df.columns:
            chainage_not_sorted = any(
                not group["chainage_m"].is_monotonic_increasing
                for _, group in df.groupby("reference_part_id", sort=False)
            )
        else:
            chainage_not_sorted = not df["chainage_m"].is_monotonic_increasing

        has_nulls = bool(df[list(self.REQUIRED_COLUMNS)].isna().any().any())

        return CoastlineNormalsValidationReport(
            count=len(df),
            has_nulls=has_nulls,
            duplicated_point_ids=int(df["point_id"].duplicated().sum()),
            invalid_tangent_count=int(invalid_tangent.sum()),
            invalid_normal_count=int(invalid_normal.sum()),
            invalid_orthogonality_count=int(invalid_orthogonality.sum()),
            invalid_side_count=int(invalid_side.sum()),
            invalid_azimuth_count=int(invalid_azimuth.sum()),
            invalid_azimuth_consistency_count=int(
                invalid_azimuth_consistency.sum()
            ),
            chainage_not_sorted=bool(chainage_not_sorted),
            min_tangent_norm=float(tangent_norm.min()),
            max_tangent_norm=float(tangent_norm.max()),
            min_normal_norm=float(normal_norm.min()),
            max_normal_norm=float(normal_norm.max()),
            max_abs_dot_product=float(np.abs(dot_product).max()),
        )

    def head_text(self, n: int = 5) -> str:
        preferred = [
            "point_id",
            "seq",
            "reference_part_id",
            "reference_distance_m",
            "chainage_m",
            "tx",
            "ty",
            "nx",
            "ny",
            "normal_azimuth_deg",
            "sea_side",
        ]
        columns = [c for c in preferred if c in self.gdf.columns]
        return self.gdf[columns].head(n).to_string(index=False)

    def debug_report(self, n: int = 5) -> str:
        summary = self.summary()
        validation = self.validate_vectors()

        return "\n".join(
            [
                f"Name: {self.name}",
                f"Count: {summary.count}",
                f"CRS: {summary.crs}",
                f"Bounds: {self.bounds}",
                f"Sea side: {summary.sea_side}",
                f"Chainage range: {summary.min_chainage_m} .. "
                f"{summary.max_chainage_m}",
                f"Mean normal azimuth: {summary.mean_normal_azimuth_deg}",
                f"Validation is_valid: {validation.is_valid}",
                f"Has nulls: {validation.has_nulls}",
                f"Duplicated point IDs: {validation.duplicated_point_ids}",
                f"Invalid tangents: {validation.invalid_tangent_count}",
                f"Invalid normals: {validation.invalid_normal_count}",
                f"Invalid orthogonality: "
                f"{validation.invalid_orthogonality_count}",
                f"Invalid side rotation: {validation.invalid_side_count}",
                f"Invalid azimuth range: {validation.invalid_azimuth_count}",
                f"Invalid azimuth consistency: "
                f"{validation.invalid_azimuth_consistency_count}",
                f"Chainage not sorted (informational): "
                f"{validation.chainage_not_sorted}",
                f"Maximum |t·n|: {validation.max_abs_dot_product}",
                "",
                f"Head({n}):",
                self.head_text(n),
            ]
        )

    def print_debug(self, n: int = 5) -> None:
        print(self.debug_report(n=n))

    def info(self) -> str:
        s = self.summary()
        return (
            f"CoastlineNormalPointSet(name={self.name!r}, count={s.count}, "
            f"crs={s.crs}, sea_side={s.sea_side})"
        )

    def __len__(self) -> int:
        return len(self.gdf)

    def __repr__(self) -> str:
        return self.info()
