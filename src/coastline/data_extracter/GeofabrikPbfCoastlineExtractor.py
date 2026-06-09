from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmium
from loguru import logger
from shapely.geometry import LineString

from src.base.BBox import BBoxExtractor, BBox

class CoastlineWayHandler(osmium.SimpleHandler):
    def __init__(self, log) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.log = log.bind(component="CoastlineWayHandler")

    def way(self, w: osmium.osm.Way) -> None:
        if w.tags.get("natural") != "coastline":
            return

        coords = []
        for node in w.nodes:
            if not node.location.valid():
                self.log.debug(f"Skip way {w.id}: invalid node location found")
                return
            coords.append((node.location.lon, node.location.lat))

        if len(coords) < 2:
            self.log.debug(f"Skip way {w.id}: less than 2 coordinates")
            return

        self.records.append(
            {
                "osm_id": w.id,
                "natural": "coastline",
                "geometry": LineString(coords),
            }
        )

        self.log.debug(f"Accepted coastline way {w.id} with {len(coords)} vertices")


class GeofabrikPbfCoastlineExtractor(BBoxExtractor):
    """
    Извлекает coastline из локального Geofabrik .osm.pbf:
    1. Делает bbox extract через osmium
    2. Читает PBF через osmium.SimpleHandler
    3. Жёстко clip-ит результат в postprocess() базового класса
    """

    def __init__(
        self,
        pbf_path: str | Path,
        bbox: BBox,
        output_path: str | Path | None = None,
        osmium_strategy: str = "complete_ways",
        osmium_bin: str = "osmium",
    ) -> None:
        super().__init__(bbox, output_path)
        self.pbf_path = Path(pbf_path)
        self.osmium_strategy = osmium_strategy
        self.osmium_bin = osmium_bin

        self.log = logger.bind(
            extractor=self.__class__.__name__,
            pbf_path=str(self.pbf_path),
            output_path=str(self.output_path) if self.output_path else "",
            bbox=self.bbox.to_osmium_bbox(),
        )

        self.log.debug(
            "Extractor initialized: "
            f"pbf_path={self.pbf_path}, "
            f"output_path={self.output_path}, "
            f"osmium_strategy={self.osmium_strategy}, "
            f"osmium_bin={self.osmium_bin}, "
            f"bbox={self.bbox.to_osmium_bbox()}"
        )

    def validate(self) -> None:
        self.log.info("Validating extractor configuration")
        super().validate()

        if not self.pbf_path.exists():
            self.log.error(f"PBF file not found: {self.pbf_path}")
            raise FileNotFoundError(f"PBF file not found: {self.pbf_path}")

        osmium_path = shutil.which(self.osmium_bin)
        if osmium_path is None:
            self.log.error(f"Command '{self.osmium_bin}' is not installed or not found in PATH")
            raise RuntimeError(
                f"Command '{self.osmium_bin}' is not installed or not found in PATH.\n"
                "Install osmium-tool and make sure the executable is available in your shell.\n\n"
                "macOS (Homebrew): brew install osmium-tool\n"
                "Ubuntu/Debian: sudo apt install osmium-tool\n\n"
                f"After installation, verify with: {self.osmium_bin} --version"
            )

        self.log.info(f"Validation successful. Found osmium binary: {osmium_path}")

    def fetch(self) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="bbox_extract_"))
        extracted_pbf = tmp_dir / "extract.osm.pbf"

        cmd = [
            self.osmium_bin,
            "extract",
            "--strategy",
            self.osmium_strategy,
            "--bbox",
            self.bbox.to_osmium_bbox(),
            str(self.pbf_path),
            "-o",
            str(extracted_pbf),
        ]

        self.log.info("Running osmium extract")
        self.log.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            self.log.exception("Failed to run osmium executable")
            raise RuntimeError(
                f"Failed to run '{self.osmium_bin}': executable not found.\n"
                "Install osmium-tool and ensure it is in PATH."
            ) from e

        self.log.debug(f"osmium return code: {result.returncode}")

        if result.stdout.strip():
            self.log.debug(f"osmium STDOUT:\n{result.stdout}")
        if result.stderr.strip():
            self.log.debug(f"osmium STDERR:\n{result.stderr}")

        if result.returncode != 0:
            self.log.error("osmium extract failed")
            raise RuntimeError(
                "osmium extract failed\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        if not extracted_pbf.exists():
            self.log.error("osmium completed but output file was not created")
            raise RuntimeError(
                "osmium finished without error, but output file was not created: "
                f"{extracted_pbf}"
            )

        self.log.info(f"Temporary extract created: {extracted_pbf}")
        return extracted_pbf

    def parse(self, raw_data: Path) -> gpd.GeoDataFrame:
        self.log.info(f"Parsing extracted PBF: {raw_data}")

        handler = CoastlineWayHandler(self.log)

        try:
            handler.apply_file(str(raw_data), locations=True)
        except Exception:
            self.log.exception("Failed while parsing PBF with osmium handler")
            raise

        self.log.info(f"Parsed coastline features: {len(handler.records)}")

        if not handler.records:
            self.log.warning("No coastline features found in extracted PBF")
            return gpd.GeoDataFrame(
                {"osm_id": [], "natural": [], "geometry": []},
                geometry="geometry",
                crs="EPSG:4326",
            )

        gdf = gpd.GeoDataFrame(
            handler.records,
            geometry="geometry",
            crs="EPSG:4326",
        )

        self.log.debug(f"GeoDataFrame created with {len(gdf)} rows")
        self.log.debug(f"GeoDataFrame columns: {list(gdf.columns)}")

        return gdf


if __name__ == "__main__":

    bbox = BBox(
        south=44.6,
        west=37.7,
        north=44.8,
        east=37.95,
    )

    try:
        app_log = logger.bind(extractor="main")

        app_log.info("Starting coastline extraction script")

        extractor = GeofabrikPbfCoastlineExtractor(
            pbf_path="../../../data/south-fed-district-260607.osm.pbf",
            bbox=bbox,
            output_path="../../../data/NovorossCoastlineOSM.geojson",
            osmium_bin="osmium",
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
