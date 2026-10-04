from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial import cKDTree
from shapely import LineString, MultiLineString, line_merge
from shapely.geometry.base import BaseGeometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.coastline.domain import CoastlineDataset

# ── Вход / выход ─────────────────────────────────────────────────────────────
INPUT_PATH = PROJECT_ROOT / "Kaliningrad" / "data" / "coastline" / "kaliningrad_OFFSET.geojson"
OUTPUT_PATH = PROJECT_ROOT / "Kaliningrad" / "data" / "coastline" / "kaliningrad_OFFSET_coastline_merged.geojson"
OUTPUT_CRS: str | None = None      # None → CRS исходного файла

# ── Параметры сшивки ─────────────────────────────────────────────────────────
SNAP_TOLERANCE_M = 10.0   # концы ближе этого расстояния считаются общей точкой (0 = без снаппинга)
DIRECTED = False         # True: не разворачивать отрезки, сшивать только конец→начало
MIN_LENGTH_M = 100.0       # отбросить обрывки короче (0 = не фильтровать)


def iter_lines(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        if len(geom.coords) >= 2:
            yield geom
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from iter_lines(part)


def snap_endpoints(lines: list[LineString], tol: float) -> list[LineString]:
    if tol <= 0 or len(lines) < 2:
        return lines

    ends = np.array([[ln.coords[0][:2], ln.coords[-1][:2]] for ln in lines]).reshape(-1, 2)
    pairs = cKDTree(ends).query_pairs(tol, output_type="ndarray")

    ds = DisjointSet(range(len(ends)))
    for i, j in pairs:
        ds.merge(int(i), int(j))

    snapped = ends.copy()
    for group in ds.subsets():
        idx = list(group)
        if len(idx) > 1:
            snapped[idx] = ends[idx].mean(axis=0)

    out = []
    for k, ln in enumerate(lines):
        coords = [c[:2] for c in ln.coords]
        coords[0] = tuple(snapped[2 * k])
        coords[-1] = tuple(snapped[2 * k + 1])
        new = LineString(coords)
        if new.length > 0:
            out.append(new)

    logger.info(f"Подтянуто концов: {len(pairs)} пар в пределах {tol} м")
    return out


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    ds = CoastlineDataset.from_geojson(main_path=INPUT_PATH, name=INPUT_PATH.stem)
    src_crs = ds.main_gdf.crs
    work_crs = ds.metric_crs
    gdf = ds.main_gdf.to_crs(work_crs)

    lines = [ln for g in gdf.geometry for ln in iter_lines(g)]
    logger.info(f"Исходных отрезков: {len(lines)}, CRS расчёта: {work_crs}")

    lines = snap_endpoints(lines, SNAP_TOLERANCE_M)
    merged = list(iter_lines(line_merge(MultiLineString(lines), directed=DIRECTED)))

    result = gpd.GeoDataFrame(geometry=merged, crs=work_crs)
    result["length_m"] = result.length.round(2)
    if MIN_LENGTH_M > 0:
        result = result[result["length_m"] >= MIN_LENGTH_M]
    result = result.sort_values("length_m", ascending=False).reset_index(drop=True)
    result.insert(0, "line_id", result.index)
    result["coastline_role"] = "main_coastline"

    total = result["length_m"].sum()
    top = result["length_m"].iloc[0] if len(result) else 0.0
    logger.info(
        f"Итоговых линий: {len(result)}; самая длинная {top:.0f} м "
        f"({100 * top / total:.1f}% общей длины {total:.0f} м)"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_crs(OUTPUT_CRS or src_crs).to_file(OUTPUT_PATH, driver="GeoJSON")
    logger.success(f"Сохранено в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()