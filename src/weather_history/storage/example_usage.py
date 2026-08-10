"""
Пример использования:
1) Загрузка скорости и направления ветра по готовому файлу точек ERA5
   (era5_coastal_grid_nodes_025deg.geojson) через WeatherPointsLoader,
   с визуализацией прогресса через tqdm.
2) Чтение временного ряда по узлу сетки, по cell_id и по произвольной координате.

Для перехода на PostgreSQL достаточно перед запуском задать переменную окружения:
    export DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/breachthebeach"
Никаких изменений в коде не требуется.
"""

from src.weather_history.storage.weather_points_loader import WeatherPointsLoader
from src.weather_history.storage.weather_repository import (
    WeatherDownloadSettings,
    WeatherRepository,
)


def main() -> None:
    settings = WeatherDownloadSettings(
        model="era5",
        grid_step=0.25,
        timezone="GMT",
    )

    loader = WeatherPointsLoader(
        points_path="/Users/mikhail_vystrchil/Documents/MY_PROGRAMMS/BreachTheBeach/src/weather_history/utils/era5_coastal_grid_nodes_025deg.geojson",
        settings=settings,
    )

    result = loader.load(
        start_date="2023-01-01",
        end_date="2024-01-31",
    )
    print("Load result:", result)

    repo = WeatherRepository(settings=settings)

    points = repo.list_grid_points()
    print(f"Grid points in DB: {len(points)}")

    first_point = points[0]
    timeseries = repo.get_timeseries_for_point(
        req_lat=first_point["req_lat"],
        req_lon=first_point["req_lon"],
        start_date="2023-01-01",
        end_date="2023-01-31",
    )
    print(f"Timeseries rows for grid point: {len(timeseries)}")
    for row in timeseries[:5]:
        print(row)  # каждая строка содержит и wind_speed_max, и wind_direction

    if first_point["cell_id"]:
        by_cell = repo.get_timeseries_by_cell_id(
            cell_id=first_point["cell_id"],
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        print(f"Timeseries rows by cell_id: {len(by_cell)}")

    point_info, nearest_series = repo.get_nearest_point_timeseries(
        lat=44.72,
        lon=37.78,
        start_date="2023-01-01",
        end_date="2023-01-31",
    )
    print("Nearest grid point:", point_info)
    print(f"Nearest point timeseries rows: {len(nearest_series)}")


if __name__ == "__main__":
    main()
