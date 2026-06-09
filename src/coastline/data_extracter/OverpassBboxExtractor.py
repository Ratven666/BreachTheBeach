from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger
from shapely.geometry import LineString

from src.base.BBox import BBox, BBoxExtractor


class OverpassCoastlineExtractor(BBoxExtractor):
    def __init__(
        self,
        bbox: BBox,
        output_path: str | Path | None = None,
        timeout: int = 180,
        endpoints: list[str] | None = None,
    ) -> None:
        super().__init__(bbox, output_path)
        self.timeout = timeout
        self.endpoints = endpoints or [
            "https://overpass-api.de/api/interpreter",
            "https://lz4.overpass-api.de/api/interpreter",
            "https://z.overpass-api.de/api/interpreter",
        ]

        self.log = logger.bind(
            extractor=self.__class__.__name__,
            bbox=self.bbox.to_overpass_bbox(),
            output_path=str(self.output_path) if self.output_path else "",
        )

        self.log.debug(
            "Extractor initialized: "
            f"bbox={self.bbox.to_overpass_bbox()}, "
            f"output_path={self.output_path}, "
            f"timeout={self.timeout}, "
            f"endpoints={self.endpoints}"
        )

    def build_query(self) -> str:
        query = f"""
[out:json][timeout:{self.timeout}];
(
  way["natural"="coastline"]({self.bbox.to_overpass_bbox()});
);
out body;
>;
out skel qt;
"""
        self.log.debug("Overpass query built successfully")
        self.log.debug(f"Query:\n{query.strip()}")
        return query

    def fetch(self) -> dict:
        headers = {
            "User-Agent": "BreachTheBeach/0.1 (Python coastline extractor)",
            "Accept": "application/json",
            "Content-Type": "text/plain; charset=utf-8",
        }

        query = self.build_query()
        last_error = None

        self.log.info("Starting fetch from Overpass API")
        self.log.info(f"Trying {len(self.endpoints)} Overpass endpoint(s)")

        for idx, endpoint in enumerate(self.endpoints, start=1):
            self.log.info(f"Trying endpoint {idx}/{len(self.endpoints)}: {endpoint}")

            try:
                response = requests.post(
                    endpoint,
                    data=query.encode("utf-8"),
                    headers=headers,
                    timeout=self.timeout + 60,
                )

                self.log.debug(
                    f"Endpoint responded with status {response.status_code}: {endpoint}"
                )

                response.raise_for_status()
                payload = response.json()

                self.log.info(f"Fetch successful from endpoint: {endpoint}")
                self.log.debug(
                    f"Received elements count: {len(payload.get('elements', []))}"
                )

                return payload

            except requests.RequestException as e:
                last_error = e
                self.log.warning(f"Endpoint failed: {endpoint} | error: {e}")

        self.log.error("All Overpass endpoints failed")
        raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")

    def parse(self, raw_data: dict) -> gpd.GeoDataFrame:
        self.log.info("Starting parse of Overpass response")

        elements = raw_data.get("elements", [])
        self.log.debug(f"Raw elements count: {len(elements)}")

        nodes: dict[int, tuple[float, float]] = {}
        ways: list[dict] = []

        for element in elements:
            if element["type"] == "node":
                nodes[element["id"]] = (element["lon"], element["lat"])
            elif element["type"] == "way":
                ways.append(element)

        self.log.debug(f"Parsed nodes count: {len(nodes)}")
        self.log.debug(f"Parsed ways count: {len(ways)}")

        records = []
        skipped_ways = 0

        for way in ways:
            coords = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]

            if len(coords) >= 2:
                records.append(
                    {
                        "osm_id": way["id"],
                        "natural": "coastline",
                        "geometry": LineString(coords),
                    }
                )
            else:
                skipped_ways += 1
                self.log.debug(
                    f"Skipped way {way.get('id')}: not enough coordinates after node join"
                )

        self.log.info(f"Constructed coastline features: {len(records)}")
        self.log.debug(f"Skipped ways: {skipped_ways}")

        if not records:
            self.log.warning("No coastline features found in Overpass response")
            return gpd.GeoDataFrame(
                {"osm_id": [], "natural": [], "geometry": []},
                geometry="geometry",
                crs="EPSG:4326",
            )

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

        self.log.info(f"GeoDataFrame created successfully with {len(gdf)} feature(s)")
        self.log.debug(f"GeoDataFrame columns: {list(gdf.columns)}")

        return gdf

if __name__ == "__main__":

    bbox = BBox(
        south=44.6,
        west=37.7,
        north=44.8,
        east=37.95,
    )

    overpass_gdf = OverpassCoastlineExtractor(
        bbox=bbox,
        output_path="output/coastline_overpass.geojson",
    ).extract()

    print("Overpass:", len(overpass_gdf))
