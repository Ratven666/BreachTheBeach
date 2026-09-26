from loguru import logger

from src.coastline.services import MainCoastlineBuilder


def main() -> None:
    builder = MainCoastlineBuilder(
        input_path="../data/NovorossCoastlineAdded.geojson",
        coastline_output_path="../nvrsk_calc/nvrsk_main_coastline.geojson",
        other_lines_output_path="../nvrsk_calc/nvrsk_other_lines.geojson",
        input_crs="EPSG:4326",
        output_crs="EPSG:4326",
        working_crs=None,
        snap_tolerance_m=3.0,
        prune_leaf_length_m=80.0,
        prune_iterations=30,
        angle_tolerance_deg=1.0,
        keep_intersection_buffer_m=0.05,
        original_vertex_buffer_m=0.05,
    )

    result = builder.build(save=True)

    logger.info(f"Main coastline features: {len(result.coastline_gdf)}")
    logger.info(f"Other lines features: {len(result.other_lines_gdf)}")
    print(result)


if __name__ == "__main__":
    main()
