from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.coastline.exporters.PointExportStrategy import PointExportStrategy


class GeoPackagePointExporter(PointExportStrategy):
    """
    Сохраняет точки как слой в GeoPackage.
    Если файл уже существует — добавляет новый слой (append=True).
    """

    def __init__(self, layer_name: str = "coastline_points") -> None:
        self.layer_name = layer_name

    def export(self, point_set: "CoastlinePointSet", output_path: str | Path) -> Path:
        output_path = Path(output_path).with_suffix(".gpkg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        point_set.gdf.to_file(
            output_path,
            layer=self.layer_name,
            driver="GPKG",
        )
        logger.info(
            f"GPKG point export: {output_path} "
            f"layer={self.layer_name!r} ({len(point_set.gdf)} pts)"
        )
        return output_path
