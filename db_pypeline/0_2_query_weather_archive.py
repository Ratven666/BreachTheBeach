from __future__ import annotations

from datetime import date
from pathlib import Path

from src.weather_history.archive import CompactWeatherRepository


DB_PATH = Path("data/db/weather_compact.db")

QUERY_LAT = 42.5
QUERY_LON = 130.5
QUERY_DATE = date(2015, 1, 1)

PERIOD_START = date(2005, 1, 1)
PERIOD_END = date(2015, 12, 31)


def main() -> None:
    with CompactWeatherRepository(DB_PATH) as repo:
        s = repo.info()
        print(f"Точек: {s['point_count']}, дней: {s['day_count']}")
        print(f"Период: {s['date_min']} — {s['date_max']}")
        print()

        point = repo.containing_grid_cell(QUERY_LAT, QUERY_LON)
        print(f"Ячейка модели для ({QUERY_LAT}, {QUERY_LON}): {point}")

        if point is None:
            print("Подходящий узел сетки не найден.")
            return

        record = repo.get_record_for_point(point, QUERY_DATE)
        if record is None:
            print("Запись не найдена.")
        else:
            print("Запись:")
            print(record)
        print()

        df = repo.get_dataframe(
            QUERY_LAT,
            QUERY_LON,
            start_date=PERIOD_START,
            end_date=PERIOD_END,
        )
        print(f"Записей в периоде {PERIOD_START}–{PERIOD_END}: {len(df)}")
        if len(df) > 0:
            print(df.describe(include="all"))
        else:
            print(df)


if __name__ == "__main__":
    main()