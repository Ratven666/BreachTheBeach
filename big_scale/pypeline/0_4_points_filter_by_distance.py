"""
Фильтрация точек GeoJSON по минимальному расстоянию.

Расстояния:
- геодезические на эллипсоиде WGS84;
- высоты Z не учитываются;
- минимальное расстояние задаётся в метрах;
- проверяется пространственная близость, не расстояние вдоль берега.

Алгоритм:
- жадный отбор в порядке исходных объектов;
- пространственный поиск через cKDTree в геоцентрических координатах;
- окончательная проверка через pyproj.Geod.inv().

Результат:
- новый GeoJSON с отобранными точками;
- исходные атрибуты сохраняются;
- исходный файл не изменяется.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


# ==================== НАСТРОЙКИ ====================

INPUT_FILE = Path(
    "/Users/mihailvystrcil/Documents/My Programms/"
    "BreachTheBeach/big_scale/data/S2CoastRussiaCropped_points_10000.geojson"
)

# None — сохранить рядом с исходным файлом:
# <имя>_min_distance_<расстояние>m.geojson
OUTPUT_FILE: Path | None = None

# Минимально допустимое расстояние между точками, метры.
MIN_DISTANCE_M = 5000.0

# Используется только при отсутствии CRS у исходного файла.
# Задавай EPSG:4326 только если координаты действительно WGS84.
INPUT_CRS_IF_MISSING: str | None = "EPSG:4326"

# Независимая итоговая проверка сохранённых точек.
VERIFY_RESULT = True

# Дополнительный запас радиуса поиска кандидатов.
# Не меняет порог удаления: удаляются только точки,
# у которых геодезическое расстояние < MIN_DISTANCE_M.
CANDIDATE_MARGIN_M = 0.0001

# Существующий выходной файл можно перезаписывать.
OVERWRITE_OUTPUT = True


# ==================== ЧТЕНИЕ ТОЧЕК ====================

def load_points(path: Path) -> gpd.GeoDataFrame:
    logger.info("Чтение GeoJSON: {}", path)

    points = gpd.read_file(path)

    if points.empty:
        raise ValueError(
            "Исходный файл не содержит объектов."
        )

    if points.crs is None:
        if INPUT_CRS_IF_MISSING is None:
            raise ValueError(
                "У исходного файла отсутствует CRS. "
                "Задай INPUT_CRS_IF_MISSING."
            )

        points = points.set_crs(
            INPUT_CRS_IF_MISSING
        )

        logger.warning(
            "CRS отсутствовала: назначена {}",
            INPUT_CRS_IF_MISSING,
        )

    nonempty = (
        points.geometry.notna()
        & ~points.geometry.is_empty
    )

    if not nonempty.all():
        raise ValueError(
            "В исходном файле есть пустые или отсутствующие "
            f"геометрии: {int((~nonempty).sum())}. "
            "Удалите их перед фильтрацией."
        )

    if not points.geometry.geom_type.eq("Point").all():
        raise TypeError(
            "Ожидался слой с отдельными Point. "
            "MultiPoint необходимо предварительно разделить."
        )

    points = points.to_crs(
        "EPSG:4326"
    ).reset_index(drop=True)

    xy = np.column_stack(
        [
            points.geometry.x.to_numpy(dtype=float),
            points.geometry.y.to_numpy(dtype=float),
        ]
    )

    if not (
        np.isfinite(xy).all()
        and np.all(np.abs(xy[:, 0]) <= 180.0)
        and np.all(np.abs(xy[:, 1]) <= 90.0)
    ):
        raise ValueError(
            "После преобразования в WGS84 получены "
            "некорректные координаты. Проверь исходную CRS."
        )

    return points


# ==================== ПРОСТРАНСТВЕННЫЙ ИНДЕКС ====================

def build_tree(lons, lats):
    """
    Переводит точки на поверхности WGS84 в геоцентрические XYZ.

    Высота принудительно равна нулю:
    фильтрация выполняется по горизонтальному расстоянию.

    Евклидова хорда между точками не длиннее пути
    по поверхности эллипсоида. Поэтому поиск по хорде
    даёт кандидатов для последующей геодезической проверки.
    """
    transformer = Transformer.from_crs(
        "EPSG:4979",
        "EPSG:4978",
        always_xy=True,
    )

    x, y, z = transformer.transform(
        lons,
        lats,
        np.zeros(len(lons), dtype=float),
    )

    xyz = np.column_stack([x, y, z])

    if not np.isfinite(xyz).all():
        raise ValueError(
            "Не удалось получить геоцентрические координаты."
        )

    return cKDTree(xyz), xyz


def geodesic_distances(
    geod,
    origin_index,
    candidate_indices,
    lons,
    lats,
):
    count = len(candidate_indices)

    _, _, distances = geod.inv(
        np.full(count, lons[origin_index]),
        np.full(count, lats[origin_index]),
        lons[candidate_indices],
        lats[candidate_indices],
    )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    if not np.isfinite(distances).all():
        raise ValueError(
            "Не удалось вычислить геодезические расстояния."
        )

    return distances


# ==================== ФИЛЬТРАЦИЯ ====================

def filter_points(lons, lats, tree, xyz):
    geod = Geod(ellps="WGS84")

    count = len(lons)

    removed = np.zeros(
        count,
        dtype=bool,
    )

    kept_indices = []

    search_radius = (
        MIN_DISTANCE_M + CANDIDATE_MARGIN_M
    )

    with tqdm(
        total=count,
        desc="Фильтрация точек",
        unit="точ.",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as progress:
        for index in range(count):
            if not removed[index]:
                kept_indices.append(index)

                candidates = np.asarray(
                    tree.query_ball_point(
                        xyz[index],
                        r=search_radius,
                        eps=0.0,
                    ),
                    dtype=np.int64,
                )

                # Проверяем только ещё не обработанные
                # и ещё не исключённые точки.
                candidates = candidates[
                    candidates > index
                ]

                candidates = candidates[
                    ~removed[candidates]
                ]

                if len(candidates):
                    distances = geodesic_distances(
                        geod,
                        index,
                        candidates,
                        lons,
                        lats,
                    )

                    too_close = candidates[
                        distances < MIN_DISTANCE_M
                    ]

                    removed[too_close] = True

            progress.update(1)

            if (index + 1) % 2000 == 0:
                progress.set_postfix(
                    kept=len(kept_indices),
                    refresh=False,
                )

    return np.asarray(
        kept_indices,
        dtype=np.int64,
    )


# ==================== ИТОГОВАЯ ПРОВЕРКА ====================

def verify_spacing(lons, lats):
    """
    Независимо проверяет, что среди отобранных точек
    нет пары с расстоянием меньше заданного.
    """
    if len(lons) < 2:
        return

    logger.info(
        "Проверка минимального расстояния результата..."
    )

    tree, xyz = build_tree(lons, lats)

    geod = Geod(ellps="WGS84")

    search_radius = (
        MIN_DISTANCE_M + CANDIDATE_MARGIN_M
    )

    for index in tqdm(
        range(len(lons)),
        desc="Проверка результата",
        unit="точ.",
        dynamic_ncols=True,
        mininterval=0.5,
    ):
        candidates = np.asarray(
            tree.query_ball_point(
                xyz[index],
                r=search_radius,
                eps=0.0,
            ),
            dtype=np.int64,
        )

        candidates = candidates[
            candidates > index
        ]

        if not len(candidates):
            continue

        distances = geodesic_distances(
            geod,
            index,
            candidates,
            lons,
            lats,
        )

        bad = np.flatnonzero(
            distances < MIN_DISTANCE_M
        )

        if len(bad):
            position = int(bad[0])

            raise RuntimeError(
                "Проверка результата не пройдена: "
                f"точки {index} и "
                f"{int(candidates[position])} "
                "находятся на расстоянии "
                f"{distances[position]:.9f} м, "
                f"порог — {MIN_DISTANCE_M} м."
            )

    logger.success(
        "Проверка пройдена: пар ближе {} м нет.",
        MIN_DISTANCE_M,
    )


# ==================== ЗАПИСЬ GEOJSON ====================

def write_geojson(points, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as stream:
            stream.write(
                '{"type":"FeatureCollection","features":[\n'
            )

            first = True

            for feature in tqdm(
                points.iterfeatures(
                    na="null",
                    drop_id=True,
                ),
                total=len(points),
                desc="Сохранение GeoJSON",
                unit="точ.",
                dynamic_ncols=True,
                mininterval=0.5,
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

        temporary.replace(path)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    if (
        not math.isfinite(MIN_DISTANCE_M)
        or MIN_DISTANCE_M <= 0
    ):
        raise ValueError(
            "MIN_DISTANCE_M должен быть положительным."
        )

    if (
        not math.isfinite(CANDIDATE_MARGIN_M)
        or CANDIDATE_MARGIN_M < 0
    ):
        raise ValueError(
            "CANDIDATE_MARGIN_M должен быть >= 0."
        )

    input_file = Path(
        INPUT_FILE
    ).expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Файл не найден:\n{input_file}"
        )

    output_file = (
        Path(OUTPUT_FILE).expanduser().resolve()
        if OUTPUT_FILE is not None
        else input_file.with_name(
            f"{input_file.stem}_min_distance_"
            f"{MIN_DISTANCE_M:g}m.geojson"
        )
    )

    if output_file == input_file:
        raise ValueError(
            "Выходной файл не должен совпадать с исходным."
        )

    temporary_file = output_file.with_name(
        output_file.name + ".tmp"
    )

    if temporary_file == input_file:
        raise ValueError(
            "Временный файл совпадает с исходным."
        )

    if output_file.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"Выходной файл уже существует:\n{output_file}"
        )

    points = load_points(input_file)

    lons = points.geometry.x.to_numpy(dtype=float)
    lats = points.geometry.y.to_numpy(dtype=float)

    logger.info(
        "Исходных точек: {}. Минимальное расстояние: {} м",
        len(points),
        MIN_DISTANCE_M,
    )

    logger.info(
        "Построение геоцентрического пространственного индекса..."
    )

    tree, xyz = build_tree(lons, lats)

    kept_indices = filter_points(
        lons,
        lats,
        tree,
        xyz,
    )

    result = points.iloc[
        kept_indices
    ].copy()

    if VERIFY_RESULT:
        verify_spacing(
            lons[kept_indices],
            lats[kept_indices],
        )

    write_geojson(
        result,
        output_file,
    )

    logger.success(
        "Готово: исходных={}, сохранено={}, удалено={}",
        len(points),
        len(result),
        len(points) - len(result),
    )

    logger.info(
        "Минимальное допустимое расстояние: {} м",
        MIN_DISTANCE_M,
    )

    logger.info(
        "Результат: {}",
        output_file,
    )


if __name__ == "__main__":
    main()