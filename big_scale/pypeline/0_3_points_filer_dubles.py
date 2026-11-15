from pathlib import Path
import json
import math


# ==================== НАСТРОЙКИ ====================

INPUT_FILE = Path(
    "/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/S2CoastRussiaCropped_points_weather_covered.geojson"
)

# None — сохранить рядом с исходным файлом:
# <имя_файла>_unique.geojson
OUTPUT_FILE = None

# None — удалять только точки с точно совпадающими координатами.
#
# Например, 8 — сравнивать координаты после округления
# до 8 знаков после запятой.
#
# Само положение сохранённых точек НЕ изменяется.
COORD_DECIMALS = None

# True — сравнивать только долготу и широту, игнорируя высоту.
# False — сравнивать все компоненты массива coordinates.
IGNORE_Z = True

# Считать долготы -180 и +180 одним меридианом.
NORMALIZE_ANTIMERIDIAN = True

# True — удалить объекты с отсутствующей/пустой геометрией.
# False — остановить обработку при их обнаружении.
DROP_EMPTY_GEOMETRIES = True

# Как часто выводить прогресс.
PROGRESS_EVERY = 100_000


# ==================== ПРОВЕРКА КООРДИНАТ ====================

def get_point_key(feature, feature_index):
    """
    Возвращает ключ сравнения координат.

    None означает отсутствующую/пустую геометрию.
    Исходный объект не изменяется.
    """
    if not isinstance(feature, dict):
        raise ValueError(
            f"Объект {feature_index}: ожидался JSON-объект."
        )

    if feature.get("type") != "Feature":
        raise ValueError(
            f"Объект {feature_index}: ожидался тип Feature."
        )

    geometry = feature.get("geometry")

    if geometry is None:
        return None

    if not isinstance(geometry, dict):
        raise ValueError(
            f"Объект {feature_index}: некорректная геометрия."
        )

    if geometry.get("type") != "Point":
        raise ValueError(
            f"Объект {feature_index}: ожидался Point, "
            f"получен {geometry.get('type')!r}."
        )

    coordinates = geometry.get("coordinates")

    if coordinates is None or coordinates == []:
        return None

    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError(
            f"Объект {feature_index}: некорректный массив coordinates."
        )

    for value in coordinates:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Объект {feature_index}: координаты должны "
                "содержать только конечные числа."
            )

    longitude = coordinates[0]
    latitude = coordinates[1]

    if not -180.0 <= longitude <= 180.0:
        raise ValueError(
            f"Объект {feature_index}: долгота {longitude} "
            "вне диапазона [-180, 180]. "
            "Скрипт ожидает координаты WGS84."
        )

    if not -90.0 <= latitude <= 90.0:
        raise ValueError(
            f"Объект {feature_index}: широта {latitude} "
            "вне диапазона [-90, 90]. "
            "Скрипт ожидает координаты WGS84."
        )

    values = (
        list(coordinates[:2])
        if IGNORE_Z
        else list(coordinates)
    )

    if COORD_DECIMALS is not None:
        values = [
            round(value, COORD_DECIMALS)
            for value in values
        ]

    # Нормализуем только ключ, не исходные координаты.
    if NORMALIZE_ANTIMERIDIAN and values[0] == 180.0:
        values[0] = -180.0

    return tuple(values)


# ==================== ОСНОВНАЯ ОБРАБОТКА ====================

def main():
    input_file = Path(INPUT_FILE).expanduser()

    output_file = (
        Path(OUTPUT_FILE).expanduser()
        if OUTPUT_FILE is not None
        else input_file.with_name(
            f"{input_file.stem}_unique.geojson"
        )
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Исходный файл не найден:\n{input_file}"
        )

    if input_file.resolve() == output_file.resolve():
        raise ValueError(
            "Выходной файл не должен совпадать с исходным."
        )

    if COORD_DECIMALS is not None:
        if (
            isinstance(COORD_DECIMALS, bool)
            or not isinstance(COORD_DECIMALS, int)
            or COORD_DECIMALS < 0
        ):
            raise ValueError(
                "COORD_DECIMALS должен быть None "
                "или неотрицательным целым числом."
            )

    if (
        isinstance(PROGRESS_EVERY, bool)
        or not isinstance(PROGRESS_EVERY, int)
        or PROGRESS_EVERY <= 0
    ):
        raise ValueError(
            "PROGRESS_EVERY должен быть положительным целым числом."
        )

    print(f"Чтение: {input_file}", flush=True)

    with input_file.open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)

    if (
        not isinstance(document, dict)
        or document.get("type") != "FeatureCollection"
    ):
        raise ValueError(
            "Ожидался GeoJSON типа FeatureCollection."
        )

    features = document.get("features")

    if not isinstance(features, list):
        raise ValueError(
            "В GeoJSON отсутствует корректный массив features."
        )

    total_count = len(features)

    print(f"Исходных объектов: {total_count:,}", flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = output_file.with_name(
        output_file.name + ".tmp"
    )

    seen = set()

    unique_count = 0
    duplicate_count = 0
    empty_count = 0

    # Сохраняем метаданные верхнего уровня, кроме bbox:
    # после удаления объектов прежний bbox может быть неактуален.
    metadata = {
        key: value
        for key, value in document.items()
        if key not in {"features", "bbox"}
    }

    try:
        with temporary_file.open("w", encoding="utf-8") as stream:
            stream.write("{")

            for key, value in metadata.items():
                json.dump(key, stream, ensure_ascii=False)
                stream.write(":")

                json.dump(
                    value,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )

                stream.write(",")

            stream.write('"features":[\n')

            first_feature = True

            for feature_index, feature in enumerate(features):
                key = get_point_key(
                    feature,
                    feature_index,
                )

                if key is None:
                    if not DROP_EMPTY_GEOMETRIES:
                        raise ValueError(
                            f"Объект {feature_index}: "
                            "отсутствующая или пустая геометрия."
                        )

                    empty_count += 1

                elif key in seen:
                    duplicate_count += 1

                else:
                    seen.add(key)

                    if not first_feature:
                        stream.write(",\n")

                    json.dump(
                        feature,
                        stream,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )

                    first_feature = False
                    unique_count += 1

                processed = feature_index + 1

                if processed % PROGRESS_EVERY == 0:
                    print(
                        f"Обработано: {processed:,} / {total_count:,}; "
                        f"уникальных: {unique_count:,}; "
                        f"дубликатов: {duplicate_count:,}",
                        flush=True,
                    )

            stream.write("\n]}\n")

        # Итоговый файл заменяется только после успешной записи.
        temporary_file.replace(output_file)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise

    print()
    print("=" * 64)
    print("РЕЗУЛЬТАТ УДАЛЕНИЯ ДУБЛИКАТОВ")
    print("=" * 64)
    print(f"Исходных объектов: {total_count:,}")
    print(f"Сохранено уникальных точек: {unique_count:,}")
    print(f"Удалено дубликатов: {duplicate_count:,}")
    print(f"Удалено пустых геометрий: {empty_count:,}")

    if COORD_DECIMALS is None:
        print("Сравнение: точное совпадение координат")
    else:
        print(
            "Сравнение: координаты, округлённые до "
            f"{COORD_DECIMALS} знаков"
        )

    print(f"Высота Z игнорируется: {IGNORE_Z}")
    print(f"Результат: {output_file.resolve()}")


if __name__ == "__main__":
    main()