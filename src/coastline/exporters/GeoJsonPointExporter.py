from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.coastline.exporters.PointExportStrategy import PointExportStrategy


class GeoJsonPointExporter(PointExportStrategy):
    """
    Сохраняет точки в один GeoJSON-файл.
    """

    def export(self, point_set: "CoastlinePointSet", output_path: str | Path) -> Path:
        output_path = Path(output_path).with_suffix(".geojson")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        point_set.gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"GeoJSON point export: {output_path} ({len(point_set.gdf)} pts)")
        return output_path
