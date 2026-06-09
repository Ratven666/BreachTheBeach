from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from loguru import logger
from shapely.geometry import box

from src.base.BBox import BBoxExtractor, BBox


class VectorCoastlineExtractor(BBoxExtractor):
    """
    Извлекает береговую линию из локального векторного файла
    (например: .shp, .gpkg, .geojson).

    Логика:
    1. Проверяет входной файл
    2. Читает слой через GeoPandas
    3. Фильтрует по bbox
    4. При необходимости фильтрует по полю natural=coastline
    5. Возвращает GeoDataFrame
    6. Дальше базовый BBoxExtractor может сделать postprocess()/save()
    """

    def __init__(
        self,
        vector_path: str | Path,
        bbox: BBox,
        output_path: str | Path | None = None,
        layer: str | None = None,
        coastline_field: str | None = "natural",
        coastline_value: str = "coastline",
        use_bbox_read_filter: bool = True,
    ) -> None:
        super().__init__(bbox, output_path)

        self.vector_path = Path(vector_path)
        self.layer = layer
        self.coastline_field = coastline_field
        self.coastline_value = coastline_value
        self.use_bbox_read_filter = use_bbox_read_filter

        self.log = logger.bind(
            extractor=self.__class__.__name__,
            vector_path=str(self.vector_path),
            output_path=str(self.output_path) if self.output_path else "",
            bbox=self.bbox.to_osmium_bbox(),
            layer=self.layer or "",
            coastline_field=self.coastline_field or "",
            coastline_value=self.coastline_value,
        )

        self.log.debug(
            "Extractor initialized: "
            f"vector_path={self.vector_path}, "
            f"output_path={self.output_path}, "
            f"layer={self.layer}, "
            f"coastline_field={self.coastline_field}, "
            f"coastline_value={self.coastline_value}, "
            f"use_bbox_read_filter={self.use_bbox_read_filter}, "
            f"bbox={self.bbox.to_osmium_bbox()}"
        )

    def validate(self) -> None:
        self.log.info("Validating extractor configuration")
        super().validate()

        if not self.vector_path.exists():
            self.log.error(f"Vector file not found: {self.vector_path}")
            raise FileNotFoundError(f"Vector file not found: {self.vector_path}")

        if not self.vector_path.is_file():
            self.log.error(f"Path is not a file: {self.vector_path}")
            raise FileNotFoundError(f"Path is not a file: {self.vector_path}")

        self.log.info("Validation successful")

    def fetch(self) -> Path:
        """
        Для локального векторного файла fetch ничего не скачивает.
        Просто возвращает путь к исходному файлу.
        """
        self.log.info(f"Using local vector source: {self.vector_path}")
        return self.vector_path

    def parse(self, raw_data: Path) -> gpd.GeoDataFrame:
        self.log.info(f"Parsing vector file: {raw_data}")

        bbox_tuple = (
            self.bbox.west,
            self.bbox.south,
            self.bbox.east,
            self.bbox.north,
        )

        read_kwargs: dict[str, Any] = {}
        if self.layer is not None:
            read_kwargs["layer"] = self.layer

        if self.use_bbox_read_filter:
            read_kwargs["bbox"] = bbox_tuple

        try:
            gdf = gpd.read_file(raw_data, **read_kwargs)
        except Exception:
            self.log.exception("Failed while reading vector file with GeoPandas")
            raise

        self.log.info(f"Loaded features: {len(gdf)}")

        if gdf.empty:
            self.log.warning("Input vector layer is empty after read_file")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if gdf.crs is None:
            self.log.warning(
                "Input vector file has no CRS. Assuming EPSG:4326 for bbox operations"
            )
            gdf = gdf.set_crs("EPSG:4326")

        bbox_geom = box(
            self.bbox.west,
            self.bbox.south,
            self.bbox.east,
            self.bbox.north,
        )
        bbox_gdf = gpd.GeoDataFrame(
            [{"geometry": bbox_geom}],
            geometry="geometry",
            crs="EPSG:4326",
        )

        if gdf.crs != bbox_gdf.crs:
            self.log.debug(f"Reprojecting bbox from {bbox_gdf.crs} to {gdf.crs}")
            bbox_gdf = bbox_gdf.to_crs(gdf.crs)

        # Если bbox-фильтр не использовался на этапе чтения, отфильтруем вручную
        if not self.use_bbox_read_filter:
            self.log.info("Applying bbox filter after read_file")
            gdf = gdf.clip(bbox_gdf)

        # Даже если bbox уже был при read_file, делаем точный clip ещё раз
        # для жёсткого обрезания геометрии по прямоугольнику
        self.log.info("Applying precise clip by bbox polygon")
        gdf = gpd.clip(gdf, bbox_gdf)

        self.log.info(f"Features after clip: {len(gdf)}")

        if gdf.empty:
            self.log.warning("No features intersect bbox after clip")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if self.coastline_field and self.coastline_field in gdf.columns:
            before = len(gdf)
            gdf = gdf[gdf[self.coastline_field] == self.coastline_value].copy()
            self.log.info(
                f"Filtered by {self.coastline_field}={self.coastline_value}: "
                f"{before} -> {len(gdf)}"
            )
        elif self.coastline_field:
            self.log.warning(
                f"Column '{self.coastline_field}' not found. "
                "Skipping attribute filter."
            )

        if gdf.empty:
            self.log.warning("No coastline features found after attribute filtering")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Оставляем только непустые геометрии
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

        if gdf.empty:
            self.log.warning("All geometries are empty after cleanup")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Нормализуем итоговую CRS в EPSG:4326
        if gdf.crs != "EPSG:4326":
            self.log.debug(f"Reprojecting result from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        self.log.debug(f"GeoDataFrame created with {len(gdf)} rows")
        self.log.debug(f"GeoDataFrame columns: {list(gdf.columns)}")

        return gdf


if __name__ == "__main__":

    from loguru import logger

    bbox = BBox(
        south=44.6,
        west=37.7,
        north=44.8,
        east=37.95,
    )

    try:
        app_log = logger.bind(extractor="main")
        app_log.info("Starting coastline extraction from vector file")

        extractor = VectorCoastlineExtractor(
            vector_path="../../../data/S2Coast2023_ShapeFile_vector/S2Coast-2023_Polyline_diss.shp",
            bbox=bbox,
            output_path="../../../data/NovorossCoastlineVectorS2Coast2023.geojson",
            layer=None,
            coastline_field="natural",
            coastline_value="coastline",
            use_bbox_read_filter=True,
        )

        coastline_gdf = extractor.extract()

        app_log.success(
            f"Extraction finished successfully. Features count: {len(coastline_gdf)}"
        )

        print(coastline_gdf.head())
        print(f"Features count: {len(coastline_gdf)}")

    except Exception:
        logger.bind(extractor="main").exception("Extractor execution failed")
        raise