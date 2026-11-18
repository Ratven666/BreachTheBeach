from pathlib import Path
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod


# ==================== НАСТРОЙКИ ====================

INPUT_FILE = Path(
    "/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/S2CoastRussiaCropped.geojson"
)

# None — сохранить рядом с исходным файлом:
# <имя_файла>_points.geojson
OUTPUT_FILE = None

# Шаг вдоль линии, метры.
STEP_M = 500.0

# Добавлять конечную точку незамкнутой линии,
# даже если до неё осталось меньше STEP_M.
INCLUDE_END_POINT = False

# Применяется только при отсутствии CRS у исходного слоя.
# Указывай "EPSG:4326" только если координаты действительно
# являются долготой и широтой WGS84.
ASSUME_CRS = None

# Число точек, обрабатываемых одним блоком.
# Точки записываются в файл постепенно.
BATCH_SIZE = 50_000

# Допуск для совпадения точки с концом линии, метры.
DISTANCE_TOLERANCE_M = 0.000001


# ==================== ГЕОМЕТРИЯ ====================

def iter_lines(geometry):
    """Возвращает непустые линейные части геометрии."""
    if geometry is None or geometry.is_empty:
        return

    if geometry.geom_type in ("LineString", "LinearRing"):
        yield geometry

    elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geometry.geoms:
            yield from iter_lines(part)


def prepare_line(line, geod):
    """
    Рассчитывает длины и начальные азимуты рёбер.

    Возвращает:
        start_lons — долготы начала ненулевых рёбер;
        start_lats — широты начала ненулевых рёбер;
        azimuths — начальные азимуты;
        edge_ends — накопленные расстояния до концов рёбер;
        total_m — общая длина линии.
    """
    coordinates = np.asarray(line.coords, dtype=float)

    lons = coordinates[:, 0]
    lats = coordinates[:, 1]

    if not np.all(np.isfinite(lons)) or not np.all(np.isfinite(lats)):
        raise ValueError("Линия содержит NaN или бесконечные координаты.")

    if np.any(np.abs(lats) > 90):
        raise ValueError(
            "Широта выходит за диапазон [-90, 90]. "
            "Проверь исходную систему координат."
        )

    azimuths, _, lengths = geod.inv(
        lons[:-1],
        lats[:-1],
        lons[1:],
        lats[1:],
    )

    azimuths = np.asarray(azimuths, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    if not np.all(np.isfinite(lengths)):
        raise ValueError("Не удалось вычислить длины рёбер.")

    # Повторяющиеся соседние вершины не участвуют в поиске ребра.
    nonzero = lengths > 0.0

    lengths = lengths[nonzero]
    azimuths = azimuths[nonzero]

    start_lons = lons[:-1][nonzero]
    start_lats = lats[:-1][nonzero]

    edge_ends = np.cumsum(lengths)
    total_m = float(edge_ends[-1]) if len(edge_ends) else 0.0

    return (
        start_lons,
        start_lats,
        azimuths,
        edge_ends,
        total_m,
    )


def iter_distance_batches(total_m, closed):
    """
    Формирует расстояния 0, STEP_M, 2*STEP_M, ...

    Для незамкнутой линии при необходимости добавляет конец.
    Для замкнутой линии исключает повтор начальной точки.
    """
    if total_m == 0.0:
        yield np.array([0.0], dtype=float)
        return

    regular_count = math.floor(total_m / STEP_M) + 1
    last_regular = None

    for start in range(0, regular_count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, regular_count)

        distances = (
            np.arange(start, stop, dtype=np.float64) * STEP_M
        )

        if closed:
            # Не создаём точку в конце кольца поверх первой.
            distances = distances[
                distances < total_m - DISTANCE_TOLERANCE_M
            ]
        else:
            distances = distances[distances <= total_m]

        if len(distances):
            last_regular = float(distances[-1])
            yield distances

    if not closed and INCLUDE_END_POINT:
        if (
            last_regular is None
            or total_m - last_regular > DISTANCE_TOLERANCE_M
        ):
            yield np.array([total_m], dtype=float)


def interpolate_batch(
    distances,
    geod,
    start_lons,
    start_lats,
    azimuths,
    edge_ends,
):
    """Вычисляет координаты точек по расстояниям вдоль линии."""
    edge_starts = np.concatenate(([0.0], edge_ends[:-1]))

    # Находим ребро, на которое попадает каждое расстояние.
    edge_indices = np.searchsorted(
        edge_ends,
        distances,
        side="right",
    )

    # Конечная точка линии должна попасть на последнее ребро.
    edge_indices = np.clip(
        edge_indices,
        0,
        len(edge_ends) - 1,
    )

    local_distances = distances - edge_starts[edge_indices]

    edge_lengths = edge_ends - edge_starts

    local_distances = np.clip(
        local_distances,
        0.0,
        edge_lengths[edge_indices],
    )

    point_lons, point_lats, _ = geod.fwd(
        start_lons[edge_indices],
        start_lats[edge_indices],
        azimuths[edge_indices],
        local_distances,
    )

    return (
        np.asarray(point_lons, dtype=float),
        np.asarray(point_lats, dtype=float),
    )


# ==================== ЗАПИСЬ GEOJSON ====================

def write_feature(
    stream,
    first_feature,
    point_id,
    feature_id,
    part_id,
    point_index,
    distance_m,
    line_length_m,
    lon,
    lat,
    has_location,
    location,
):
    """Записывает одну точку в поток GeoJSON."""
    properties = {
        "point_id": point_id,
        "feature_id": feature_id,
        "part_id": part_id,
        "point_index": point_index,
        "distance_m": float(distance_m),
        "line_length_m": float(line_length_m),
    }

    if has_location:
        properties["Location"] = location

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)],
        },
        "properties": properties,
    }

    if not first_feature:
        stream.write(",\n")

    json.dump(
        feature,
        stream,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


# ==================== ОСНОВНАЯ ОБРАБОТКА ====================

def main():
    input_file = Path(INPUT_FILE).expanduser()

    output_file = (
        Path(OUTPUT_FILE).expanduser()
        if OUTPUT_FILE is not None
        else input_file.with_name(
            f"{input_file.stem}_points.geojson"
        )
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Исходный файл не найден:\n{input_file}"
        )

    if input_file.resolve() == output_file.resolve():
        raise ValueError(
            "Исходный файл и выходной файл не должны совпадать."
        )

    if not math.isfinite(STEP_M) or STEP_M <= 0:
        raise ValueError("STEP_M должен быть положительным числом.")

    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE должен быть положительным целым.")

    print(f"Чтение: {input_file}", flush=True)

    data = gpd.read_file(input_file)

    if data.crs is None:
        if ASSUME_CRS is None:
            raise ValueError(
                "У исходного слоя отсутствует CRS. "
                "Укажи ASSUME_CRS, если исходная СК известна."
            )

        data = data.set_crs(ASSUME_CRS)

    print(f"Исходная CRS: {data.crs}", flush=True)

    # Входной слой должен иметь одну корректную исходную CRS.
    data = data.to_crs("EPSG:4326")

    geod = Geod(ellps="WGS84")
    has_location = "Location" in data.columns

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Сначала пишем во временный файл.
    # Итоговый файл заменяется только после успешной обработки.
    temporary_file = output_file.with_name(
        output_file.name + ".tmp"
    )

    point_count = 0
    line_count = 0
    zero_length_count = 0
    total_length_m = 0.0
    first_feature = True

    try:
        with temporary_file.open("w", encoding="utf-8") as stream:
            stream.write(
                '{"type":"FeatureCollection","features":[\n'
            )

            for feature_id, (_, row) in enumerate(data.iterrows()):
                location = None

                if has_location:
                    value = row["Location"]
                    if pd.notna(value):
                        location = str(value)

                for part_id, line in enumerate(iter_lines(row.geometry)):
                    try:
                        (
                            start_lons,
                            start_lats,
                            azimuths,
                            edge_ends,
                            total_m,
                        ) = prepare_line(line, geod)

                    except ValueError as error:
                        raise ValueError(
                            f"Объект {feature_id}, часть {part_id}: "
                            f"{error}"
                        ) from error

                    line_count += 1
                    total_length_m += total_m

                    if total_m == 0.0:
                        zero_length_count += 1

                        lon, lat = line.coords[0][:2]

                        write_feature(
                            stream=stream,
                            first_feature=first_feature,
                            point_id=point_count,
                            feature_id=feature_id,
                            part_id=part_id,
                            point_index=0,
                            distance_m=0.0,
                            line_length_m=0.0,
                            lon=lon,
                            lat=lat,
                            has_location=has_location,
                            location=location,
                        )

                        first_feature = False
                        point_count += 1
                        continue

                    point_index = 0

                    for distances in iter_distance_batches(
                        total_m,
                        closed=line.is_closed,
                    ):
                        point_lons, point_lats = interpolate_batch(
                            distances=distances,
                            geod=geod,
                            start_lons=start_lons,
                            start_lats=start_lats,
                            azimuths=azimuths,
                            edge_ends=edge_ends,
                        )

                        for distance_m, lon, lat in zip(
                            distances,
                            point_lons,
                            point_lats,
                        ):
                            write_feature(
                                stream=stream,
                                first_feature=first_feature,
                                point_id=point_count,
                                feature_id=feature_id,
                                part_id=part_id,
                                point_index=point_index,
                                distance_m=distance_m,
                                line_length_m=total_m,
                                lon=lon,
                                lat=lat,
                                has_location=has_location,
                                location=location,
                            )

                            first_feature = False
                            point_count += 1
                            point_index += 1

                    if line_count % 1000 == 0:
                        print(
                            f"Обработано линий: {line_count:,}; "
                            f"создано точек: {point_count:,}",
                            flush=True,
                        )

            stream.write("\n]}\n")

        temporary_file.replace(output_file)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise

    print()
    print(f"Обработано линейных частей: {line_count:,}")
    print(f"Из них нулевой длины: {zero_length_count:,}")
    print(f"Суммарная длина: {total_length_m / 1000:,.3f} км")
    print(f"Заданный шаг: {STEP_M:,.3f} м")
    print(f"Создано точек: {point_count:,}")
    print(f"Результат: {output_file.resolve()}")


if __name__ == "__main__":
    main()