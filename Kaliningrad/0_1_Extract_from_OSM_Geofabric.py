from loguru import logger

from base.BBox import BBox
from coastline.data_extracter.GeofabrikPbfCoastlineExtractor import GeofabrikPbfCoastlineExtractor

bbox = BBox(
    south=53.9,
    west=19,
    north=55.6,
    east=21.7,
)

try:
    app_log = logger.bind(extractor="main")

    app_log.info("Starting coastline extraction script")

    extractor = GeofabrikPbfCoastlineExtractor(
        pbf_path="data/northwestern-fed-district-261002.osm.pbf",
        bbox=bbox,
        output_path="data/KaliningradOSM.geojson",
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
