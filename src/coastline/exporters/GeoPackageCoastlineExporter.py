from __future__ import annotations

from pathlib import Path

from src.coastline.exporters.CoastlineExportStrategy import CoastlineExportStrategy


class GeoPackageCoastlineExporter(CoastlineExportStrategy):
    """
    Экспортирует в GeoPackage.
    Создаются слои:
    - main_coastline
    - other_coastline
    """

    def __init__(
        self,
        main_layer_name: str = "main_coastline",
        other_layer_name: str = "other_coastline",
    ) -> None:
        self.main_layer_name = main_layer_name
        self.other_layer_name = other_layer_name

    def export(self, dataset: "CoastlineDataset", output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() != ".gpkg":
            output_path = output_path.with_suffix(".gpkg")

        dataset.main_gdf.to_file(
            output_path,
            layer=self.main_layer_name,
            driver="GPKG",
        )
        dataset.other_gdf.to_file(
            output_path,
            layer=self.other_layer_name,
            driver="GPKG",
        )

        dataset.log.info(
            f"GeoPackage exported: path={output_path}, "
            f"layers=({self.main_layer_name}, {self.other_layer_name})"
        )
        return output_path
