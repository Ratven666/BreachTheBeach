from __future__ import annotations

"""
Примеры обращения к сохранённой батиметрии.

Запуск:

    python db_pypeline/4_2_bathymetry_example.py

Переменная окружения:

    BATHY_MANIFEST
        Путь к manifest.json.
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.bathymetry.archive import (
    BathymetryArchive,
)


MANIFEST_PATH = Path(
    os.getenv(
        "BATHY_MANIFEST",
        "data/bathymetry/manifest.json",
    )
)


def main() -> None:
    archive = BathymetryArchive(
        MANIFEST_PATH
    )

    print(
        "Point source id:",
        archive.point_source_id,
    )

    print(
        "Archived points:",
        len(archive.point_ids),
    )

    first_point_id = (
        archive.point_ids[0]
    )

    point_metadata = (
        archive.point_metadata(
            first_point_id
        )
    )

    print(
        "First point:",
        point_metadata,
    )

    # radius_m=None означает использование максимального
    # fetch_length_m этой точки, прочитанного из БД.
    point_bathymetry = (
        archive.for_point_id(
            first_point_id,
            radius_m=None,
            n_steps=200,
        )
    )

    profile = (
        point_bathymetry.get_profile(
            direction=225
        )
    )

    print(
        f"point_id={first_point_id}"
    )

    print(
        "profile length="
        f"{profile.distances_m[-1]:.1f} m"
    )

    finite_count = int(
        (
            profile.depths_m
            == profile.depths_m
        ).sum()
    )

    print(
        "finite underwater samples="
        f"{finite_count}"
    )

    print(
        "first depths:",
        profile.depths_m[:10],
    )

    # Поиск по координатам. Именно этот интерфейс использует
    # существующий WaveClimateBatchProcessor.
    same_point_service = (
        archive.for_point(
            lon=float(
                point_metadata["lon"]
            ),
            lat=float(
                point_metadata["lat"]
            ),
            n_steps=200,
        )
    )

    assert (
        same_point_service.point_id
        == first_point_id
    )

    profile_90 = (
        same_point_service.get_profile(
            direction=90
        )
    )

    print(
        "90-degree profile length="
        f"{profile_90.distances_m[-1]:.1f} m"
    )

    # Интеграция с существующим пакетным расчётом:
    #
    # processor.export(
    #     points_path=POINTS_PATH,
    #     fetch_csv_path=FETCH_CSV,
    #     weather_csv_path=WEATHER_CSV,
    #     bathymetry_service=archive,
    #     daily_output_path=DAILY_OUT,
    #     summary_output_path=SUMMARY_OUT,
    # )
    #
    # WaveClimateBatchProcessor вызовет:
    #
    #     archive.for_point(lon, lat)
    #
    # а WaveClimateService:
    #
    #     point_service.get_profile(direction)
    #     profile.depths_m


if __name__ == "__main__":
    main()
