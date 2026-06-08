from __future__ import annotations

from pathlib import Path

from src.coastline.exporters.CoastlineExportStrategy import CoastlineExportStrategy


class GeoJsonCoastlineExporter(CoastlineExportStrategy):
    """
    Экспортирует в GeoJSON.
    Создаются два файла:
    - *_main.geojson
    - *_other.geojson
    """

    def export(self, dataset: "CoastlineDataset", output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stem = output_path.stem
        suffix = output_path.suffix.lower()

        if suffix != ".geojson":
            output_path = output_path.with_suffix(".geojson")
            stem = output_path.stem

        main_path = output_path.with_name(f"{stem}_main.geojson")
        other_path = output_path.with_name(f"{stem}_other.geojson")

        dataset.main_gdf.to_file(main_path, driver="GeoJSON")
        dataset.other_gdf.to_file(other_path, driver="GeoJSON")

        dataset.log.info(f"GeoJSON exported: main={main_path}, other={other_path}")
        return output_path.parent
