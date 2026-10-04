from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial import cKDTree
from shapely import LineString, MultiLineString, line_merge, unary_union
from shapely.geometry.base import BaseGeometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.coastline.domain import CoastlineDataset

# ── Входные слои ─────────────────────────────────────────────────────────────
LAYER_A = PROJECT_ROOT / "Kaliningrad" / "data" / "coastline" / "KaliningradOSM_main_coastline.geojson"
LAYER_B = PROJECT_ROOT / "Kaliningrad" / "data" / "coastline" / "KaliningradOSM_other_lines.geojson"

# ── Выход ────────────────────────────────────────────────────────────────────
OUTPUT_PATH = PROJECT_ROOT / "Kaliningrad" / "data" / "coastline" / "KaliningradOSM_all_lines.geojson"
OUTPUT_CRS = "EPSG:4326"

# ── Параметры сшивки (метры) ─────────────────────────────────────────────────
SNAP_TOLERANCE_M = 10.0     # концы ближе этого расстояния считаются одной точкой
NODE_INTERSECTIONS = True # True: разрезать в пересечениях и убрать дубли перекрытий
MIN_LENGTH_M = 100.0         # отбросить обрывки короче (0 = не фильтровать)


def iter_lines(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        if len(geom.coords) >= 2:
            yield geom
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from iter_lines(part)


def load_lines(path: Path, name: str) -> gpd.GeoDataFrame:
    ds = CoastlineDataset.from_geojson(main_path=path, name=name)
    gdf = ds.main_gdf.to_crs(ds.metric_crs)
    logger.info(f"{name}: {len(gdf)} объектов, metric CRS = {ds.metric_crs}")
    return gdf[["geometry"]]


def snap_endpoints(lines: list[LineString], tol: float) -> list[LineString]:
    if tol <= 0 or not lines:
        return lines

    ends = np.array([[ln.coords[0], ln.coords[-1]] for ln in lines])[:, :, :2].reshape(-1, 2)
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

    logger.info(f"Склеено концов: {len(pairs)} пар в пределах {tol} м")
    return out


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    gdf_a = load_lines(LAYER_A, "layer_a")
    work_crs = gdf_a.crs
    gdf_b = load_lines(LAYER_B, "layer_b").to_crs(work_crs)

    lines = [ln for g in pd.concat([gdf_a, gdf_b]).geometry for ln in iter_lines(g)]
    logger.info(f"Исходных линий: {len(lines)}")

    lines = snap_endpoints(lines, SNAP_TOLERANCE_M)

    geom = unary_union(lines) if NODE_INTERSECTIONS else MultiLineString(lines)
    merged = list(iter_lines(line_merge(geom)))

    result = gpd.GeoDataFrame(geometry=merged, crs=work_crs)
    result["length_m"] = result.length.round(2)
    if MIN_LENGTH_M > 0:
        result = result[result["length_m"] >= MIN_LENGTH_M]
    result = result.sort_values("length_m", ascending=False).reset_index(drop=True)
    result.insert(0, "line_id", result.index)

    logger.info(f"Итоговых линий: {len(result)}, самая длинная: {result['length_m'].max():.0f} м")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_crs(OUTPUT_CRS).to_file(OUTPUT_PATH, driver="GeoJSON")
    logger.success(f"Сохранено в {OUTPUT_PATH} ({OUTPUT_CRS})")


if __name__ == "__main__":
    main()