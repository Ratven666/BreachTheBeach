from pathlib import Path
import json
import sqlite3
from contextlib import closing

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree


# ==================== НАСТРОЙКИ ====================

INPUT_FILE = Path("/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/S2CoastRussiaCropped_points.geojson"
)

DB_PATH = Path(
    "/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/db/weather_compact.db"
)

# None — сохранить рядом с исходным файлом:
# <имя_файла>_weather_covered.geojson
OUTPUT_FILE = None

# Шаг метеосетки согласно repository.py.
GRID_STEP_DEG = 0.25

# Масштаб хранения координат согласно schema.py.
COORD_SCALE = 1000

# True — учитывать только узлы, имеющие хотя бы одну запись
# в weather_days.
REQUIRE_WEATHER_RECORDS = True

# Дополнительное условие:
# True — у узла должна быть хотя бы одна запись, в которой
# одновременно заполнены wind_speed_max_i и wind_direction_i.
#
# При True проверка наличия записей применяется независимо
# от REQUIRE_WEATHER_RECORDS.
REQUIRE_WIND_PAIR = False

# Используется только при отсутствии CRS исходного GeoJSON.
# Укажи "EPSG:4326" только если исходные координаты —
# действительно долгота и широта WGS84.
ASSUME_CRS = None

# Размер блока проверки береговых точек.
BATCH_SIZE = 100_000

# Допуск на границе ячейки, градусы.
EPS = 1e-9


# ==================== ЧТЕНИЕ МЕТЕОСЕТКИ ====================

def load_weather_grid(db_path: Path):
    """
    Читает узлы метеосетки.

    Возвращает:
        grid_ids — идентификаторы узлов;
        grid_xy — массив [долгота, широта] в градусах.
    """
    if not db_path.is_file():
        raise FileNotFoundError(
            f"База данных не найдена:\n{db_path}"
        )

    uri = db_path.resolve().as_uri() + "?mode=ro"

    with closing(
        sqlite3.connect(uri, uri=True, timeout=60)
    ) as connection:
        connection.execute("PRAGMA query_only = ON")

        required_tables = {
            "weather_grid_points",
            "weather_days",
        }

        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        missing_tables = required_tables - existing_tables

        if missing_tables:
            raise ValueError(
                "База не соответствует ожидаемой схеме "
                "компактного метеоархива. "
                f"Отсутствуют таблицы: {sorted(missing_tables)}"
            )

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(weather_grid_points)"
            )
        }

        if not {"id", "lat_i", "lon_i"}.issubset(columns):
            raise ValueError(
                "В weather_grid_points отсутствуют "
                "поля id, lat_i или lon_i."
            )

        condition = ""

        if REQUIRE_WIND_PAIR:
            condition = """
                WHERE EXISTS (
                    SELECT 1
                    FROM weather_days AS d
                    WHERE d.point_id = p.id
                      AND d.wind_speed_max_i IS NOT NULL
                      AND d.wind_direction_i IS NOT NULL
                )
            """

        elif REQUIRE_WEATHER_RECORDS:
            condition = """
                WHERE EXISTS (
                    SELECT 1
                    FROM weather_days AS d
                    WHERE d.point_id = p.id
                )
            """

        query = f"""
            SELECT p.id, p.lon_i, p.lat_i
            FROM weather_grid_points AS p
            {condition}
            ORDER BY p.id
        """

        rows = connection.execute(query).fetchall()

    if not rows:
        raise ValueError(
            "Не найдено узлов метеосетки, соответствующих "
            "настройкам фильтрации."
        )

    grid_ids = np.array(
        [row[0] for row in rows],
        dtype=np.int64,
    )

    grid_xy = np.array(
        [[row[1], row[2]] for row in rows],
        dtype=np.float64,
    ) / COORD_SCALE

    if not np.all(np.isfinite(grid_xy)):
        raise ValueError("Метеосетка содержит некорректные координаты.")

    if np.any(np.abs(grid_xy[:, 1]) > 90):
        raise ValueError("Широты метеосетки выходят за диапазон [-90, 90].")

    # Нормализуем долготу в диапазон [-180, 180).
    grid_xy[:, 0] = (
        (grid_xy[:, 0] + 180.0) % 360.0
    ) - 180.0

    return grid_ids, grid_xy


# ==================== ПРОСТРАНСТВЕННЫЙ ИНДЕКС ====================

def build_grid_tree(grid_xy):
    """
    Создаёт индекс узлов.

    Копии с долготами ±360° нужны для корректной проверки
    точек и ячеек около антимеридиана.
    """
    left_copy = grid_xy.copy()
    left_copy[:, 0] -= 360.0

    right_copy = grid_xy.copy()
    right_copy[:, 0] += 360.0

    extended_xy = np.vstack(
        [grid_xy, left_copy, right_copy]
    )

    return cKDTree(extended_xy)


def get_coverage_mask(points_xy, tree):
    """
    Проверяет попадание в объединение квадратных ячеек
    вокруг существующих узлов.

    p=inf означает максимум из абсолютных отклонений
    по долготе и широте.
    """
    keep = np.zeros(len(points_xy), dtype=bool)

    valid = (
        np.isfinite(points_xy).all(axis=1)
        & (np.abs(points_xy[:, 1]) <= 90.0)
    )

    valid_indices = np.flatnonzero(valid)
    half_step = GRID_STEP_DEG / 2.0

    for start in range(0, len(valid_indices), BATCH_SIZE):
        indices = valid_indices[start:start + BATCH_SIZE]

        query_xy = points_xy[indices].copy()

        query_xy[:, 0] = (
            (query_xy[:, 0] + 180.0) % 360.0
        ) - 180.0

        distances, _ = tree.query(
            query_xy,
            k=1,
            p=np.inf,
            workers=-1,
        )

        keep[indices] = distances <= half_step + EPS

        print(
            f"Проверено точек: "
            f"{min(start + BATCH_SIZE, len(valid_indices)):,}"
            f" / {len(valid_indices):,}",
            flush=True,
        )

    return keep, valid


# ==================== ЗАПИСЬ GEOJSON ====================

def write_geojson(data, output_file):
    """
    Сохраняет GeoJSON, включая случай пустого результата.

    Запись сначала выполняется во временный файл.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = output_file.with_name(
        output_file.name + ".tmp"
    )

    try:
        with temporary_file.open("w", encoding="utf-8") as stream:
            stream.write(
                '{"type":"FeatureCollection","features":[\n'
            )

            first = True

            for feature in data.iterfeatures(
                na="null",
                drop_id=True,
            ):
                if not first:
                    stream.write(",\n")

                json.dump(
                    feature,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )

                first = False

            stream.write("\n]}\n")

        temporary_file.replace(output_file)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


# ==================== ОСНОВНАЯ ОБРАБОТКА ====================

def main():
    input_file = Path(INPUT_FILE).expanduser()
    db_path = Path(DB_PATH).expanduser()

    output_file = (
        Path(OUTPUT_FILE).expanduser()
        if OUTPUT_FILE is not None
        else input_file.with_name(
            f"{input_file.stem}_weather_covered.geojson"
        )
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"GeoJSON не найден:\n{input_file}"
        )

    protected_paths = {
        input_file.resolve(),
        db_path.resolve(),
    }

    if output_file.resolve() in protected_paths:
        raise ValueError(
            "Выходной файл не должен совпадать "
            "с исходным GeoJSON или базой данных."
        )

    if not np.isfinite(GRID_STEP_DEG) or GRID_STEP_DEG <= 0:
        raise ValueError(
            "GRID_STEP_DEG должен быть положительным числом."
        )

    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE должен быть положительным целым числом."
        )

    print(f"Чтение метеосетки: {db_path}", flush=True)

    grid_ids, grid_xy = load_weather_grid(db_path)

    print(f"Подходящих узлов: {len(grid_ids):,}")
    print(f"Шаг сетки: {GRID_STEP_DEG}°")
    print("Построение пространственного индекса...", flush=True)

    tree = build_grid_tree(grid_xy)

    print(f"Чтение береговых точек: {input_file}", flush=True)

    points = gpd.read_file(input_file)
    input_count = len(points)

    if points.crs is None:
        if ASSUME_CRS is None:
            raise ValueError(
                "У исходного слоя отсутствует CRS. "
                "Укажи ASSUME_CRS, если исходная СК известна."
            )

        points = points.set_crs(ASSUME_CRS)

    print(f"Исходная CRS: {points.crs}")
    print(f"Исходных объектов: {input_count:,}")

    nonempty = (
        points.geometry.notna()
        & ~points.geometry.is_empty
    )

    wrong_type = (
        nonempty
        & (points.geometry.geom_type != "Point")
    )

    if wrong_type.any():
        raise ValueError(
            "Ожидался слой только с геометрией Point. "
            f"Других непустых геометрий: {int(wrong_type.sum()):,}. "
            "MultiPoint необходимо предварительно разделить "
            "на отдельные точки."
        )

    # Отсутствующие и пустые геометрии не сохраняем.
    usable = points.loc[nonempty].copy()
    usable = usable.to_crs("EPSG:4326")

    points_xy = np.column_stack(
        [
            usable.geometry.x.to_numpy(),
            usable.geometry.y.to_numpy(),
        ]
    )

    print("Проверка покрытия...", flush=True)

    keep, valid_coordinates = get_coverage_mask(
        points_xy,
        tree,
    )

    result = usable.iloc[np.flatnonzero(keep)].copy()

    print("Сохранение GeoJSON...", flush=True)

    write_geojson(result, output_file)

    empty_count = input_count - len(usable)
    invalid_coordinate_count = int(
        (~valid_coordinates).sum()
    )
    outside_count = int(
        (valid_coordinates & ~keep).sum()
    )

    print()
    print("=" * 64)
    print("РЕЗУЛЬТАТ ФИЛЬТРАЦИИ")
    print("=" * 64)
    print(f"Всего исходных объектов: {input_count:,}")
    print(f"Узлов метеосетки использовано: {len(grid_ids):,}")
    print(f"Сохранено точек: {len(result):,}")
    print(f"Удалено всего: {input_count - len(result):,}")
    print(f"  Вне покрытия: {outside_count:,}")
    print(f"  Пустые/отсутствующие геометрии: {empty_count:,}")
    print(f"  Некорректные координаты: {invalid_coordinate_count:,}")

    if input_count:
        print(
            f"Доля сохранённых: "
            f"{100.0 * len(result) / input_count:.2f}%"
        )

    print(f"Результат: {output_file.resolve()}")


if __name__ == "__main__":
    main()