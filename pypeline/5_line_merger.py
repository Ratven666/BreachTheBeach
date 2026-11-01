from __future__ import annotations

from src.coastline.domain import CoastlineDataset

if __name__ == "__main__":
    from loguru import logger

    OUTER_MAIN_PATH = r"../data/NVRSK_BlackSeaCoastlineS2Coast2023.geojson"
    # OUTER_OTHER_PATH = r"data/nvrsk_black_sea_other_lines_fast.geojson"

    INNER_MAIN_PATH = r"../nvrsk_calc/nvrsk_main_coastline.geojson"
    INNER_OTHER_PATH = r"../nvrsk_calc/nvrsk_other_lines.geojson"

    OUTPUT_MAIN_PATH = r"../nvrsk_calc/for_example/merged_main.geojson"
    OUTPUT_OTHER_PATH = r"../nvrsk_calc/for_example/merged_other.geojson"
    OUTPUT_COMBINED_PATH = r"../nvrsk_calc/merged_dataset.geojson"

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True)

    ds_outer = CoastlineDataset.from_geojson(
        main_path=OUTER_MAIN_PATH,
        # other_path=OUTER_OTHER_PATH,
        name="outer_dataset",
    )

    ds_inner = CoastlineDataset.from_geojson(
        main_path=INNER_MAIN_PATH,
        other_path=INNER_OTHER_PATH,
        name="inner_dataset",
    )

    merged = ds_outer.merge_with_replacement(
        ds_inner,
        close_gaps=True,
        snap_tolerance=2.0,
        max_gap_distance=100.0,
        name="merged_dataset",
    )

    merged.print_summary()

    merged.export_split_geojson(
        main_output_path=OUTPUT_MAIN_PATH,
        other_output_path=OUTPUT_OTHER_PATH,
    )

    merged.export_combined_geojson(OUTPUT_COMBINED_PATH)

    logger.success("Export completed")
