from __future__ import annotations

from pathlib import Path

from src.coastline.exporters.CoastlineExportStrategy import CoastlineExportStrategy


class GeoPackageCoastlineExporter(CoastlineExportStrategy):
    """
    Экспорт линий в один GeoPackage с двумя слоями.
    """

    def __init__(
        self,
        main_layer: str = "main_coastline",
        other_layer: str = "other_coastline",
    ) -> None:
        self.main_layer = main_layer
        self.other_layer = other_layer

    def export(self, dataset: "CoastlineDataset", output_path: str | Path) -> Path:
        output_path = Path(output_path).with_suffix(".gpkg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset.main_gdf.to_file(output_path, layer=self.main_layer, driver="GPKG")
        dataset.other_gdf.to_file(output_path, layer=self.other_layer, driver="GPKG")

        dataset._log.info(f"GPKG export: {output_path}")
        return output_path
