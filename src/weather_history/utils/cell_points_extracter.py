import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


# ---------------------------------------------------------------------
# ПАРАМЕТРЫ
# ---------------------------------------------------------------------
INPUT_FILE = Path("/Users/mikhail_vystrchil/Documents/MY_PROGRAMMS/BreachTheBeach/data/NVRSK_BlackSeaCoastlineS2Coast2023.geojson")

OUTPUT_CELLS = Path("era5_coastal_cells_025deg.geojson")
OUTPUT_POINTS = Path("era5_coastal_grid_nodes_025deg.geojson")

STEP = 0.25          # Шаг сетки ERA5, градусы
HALF_STEP = STEP / 2 # Половина размера ячейки: 0.125°


# ---------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------
def ceil_to_grid(value: float, step: float) -> float:
    """Округляет координату вверх до ближайшего узла регулярной сетки."""
    return math.ceil(value / step) * step


def floor_to_grid(value: float, step: float) -> float:
    """Округляет координату вниз до ближайшего узла регулярной сетки."""
    return math.floor(value / step) * step


# ---------------------------------------------------------------------
# ЧТЕНИЕ БЕРЕГОВОЙ ЛИНИИ
# ---------------------------------------------------------------------
coast = gpd.read_file(INPUT_FILE)

# GeoJSON использует порядок координат longitude, latitude.
# EPSG:4326 подходит для сетки ERA5.
if coast.crs is None:
    coast = coast.set_crs("EPSG:4326")
else:
    coast = coast.to_crs("EPSG:4326")

# Оставляем только непустые геометрии
coast = coast[
    coast.geometry.notna() &
    ~coast.geometry.is_empty
].copy()

# Объединённая береговая линия
coastline = coast.geometry.unary_union

# Границы береговой линии
min_lon, min_lat, max_lon, max_lat = coastline.bounds


# ---------------------------------------------------------------------
# ПОСТРОЕНИЕ УЗЛОВ И ЯЧЕЕК ERA5
# ---------------------------------------------------------------------
# Ячейка узла (lon, lat) имеет границы:
# [lon - 0.125, lon + 0.125] × [lat - 0.125, lat + 0.125].
#
# Поэтому ищем все центры ячеек, потенциально пересекающие
# bounding box береговой линии.

first_lon = ceil_to_grid(min_lon - HALF_STEP, STEP)
last_lon = floor_to_grid(max_lon + HALF_STEP, STEP)

first_lat = ceil_to_grid(min_lat - HALF_STEP, STEP)
last_lat = floor_to_grid(max_lat + HALF_STEP, STEP)

n_cols = round((last_lon - first_lon) / STEP) + 1
n_rows = round((last_lat - first_lat) / STEP) + 1

cells = []

for col in range(n_cols):
    center_lon = round(first_lon + col * STEP, 8)

    for row in range(n_rows):
        center_lat = round(first_lat + row * STEP, 8)

        # Границы ячейки, центрированной на ERA5-узле
        lon_min = round(center_lon - HALF_STEP, 8)
        lon_max = round(center_lon + HALF_STEP, 8)
        lat_min = round(center_lat - HALF_STEP, 8)
        lat_max = round(center_lat + HALF_STEP, 8)

        cells.append(
            {
                "era5_lon": center_lon,
                "era5_lat": center_lat,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "col": col,
                "row": row,
                "geometry": box(lon_min, lat_min, lon_max, lat_max),
            }
        )

era5_grid = gpd.GeoDataFrame(cells, crs="EPSG:4326")


# ---------------------------------------------------------------------
# ВЫБОР ЯЧЕЕК, ПЕРЕСЕКАЕМЫХ БЕРЕГОВОЙ ЛИНИЕЙ
# ---------------------------------------------------------------------
coastal_cells = era5_grid[era5_grid.intersects(coastline)].copy()

# Идентификатор ячейки через координаты узла ERA5
coastal_cells["cell_id"] = (
    coastal_cells["era5_lon"].map(lambda x: f"{x:.2f}")
    + "_"
    + coastal_cells["era5_lat"].map(lambda y: f"{y:.2f}")
)


# ---------------------------------------------------------------------
# СОЗДАНИЕ ТОЧЕК В УЗЛАХ ERA5
# ---------------------------------------------------------------------
era5_points = coastal_cells.copy()

era5_points["geometry"] = gpd.points_from_xy(
    era5_points["era5_lon"],
    era5_points["era5_lat"],
    crs="EPSG:4326",
)

era5_points = era5_points[
    [
        "cell_id",
        "era5_lon",
        "era5_lat",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "col",
        "row",
        "geometry",
    ]
]


# ---------------------------------------------------------------------
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ---------------------------------------------------------------------
coastal_cells.to_file(OUTPUT_CELLS, driver="GeoJSON")
era5_points.to_file(OUTPUT_POINTS, driver="GeoJSON")

print(f"Количество береговых ячеек ERA5: {len(coastal_cells)}")
print(f"Ячейки сохранены: {OUTPUT_CELLS}")
print(f"Узлы ERA5 сохранены: {OUTPUT_POINTS}")