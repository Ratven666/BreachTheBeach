from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import geopandas as gpd
import networkx as nx
from shapely import node as shapely_node, snap, unary_union
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge


@dataclass
class CoastlineBuildResult:
    coastline_gdf: gpd.GeoDataFrame
    other_lines_gdf: gpd.GeoDataFrame


class MainCoastlineBuilder:
    """
    Выделение главной береговой линии с сохранением исходных вершин
    и последующим спрямлением искусственно добавленных сегментов.

    Новая логика:
    - сохраняем исходные вершины до любых модификаций;
    - строим топологически корректную сеть;
    - выделяем главную трассу;
    - после этого удаляем лишние точки, оставляя только:
        * исходные вершины,
        * точки пересечений,
        * реальные точки излома.
    """

    _COORD_ROUND = 3

    def __init__(
        self,
        input_path: str | Path,
        coastline_output_path: str | Path | None = None,
        other_lines_output_path: str | Path | None = None,
        input_crs: str = "EPSG:4326",
        output_crs: str = "EPSG:4326",
        working_crs: str | None = None,
        snap_tolerance_m: float = 3.0,
        prune_leaf_length_m: float = 80.0,
        prune_iterations: int = 30,
        angle_tolerance_deg: float = 1.0,
        keep_intersection_buffer_m: float = 0.05,
        original_vertex_buffer_m: float = 0.05,
    ) -> None:
        self.input_path = Path(input_path)
        self.coastline_output_path = (
            Path(coastline_output_path) if coastline_output_path else None
        )
        self.other_lines_output_path = (
            Path(other_lines_output_path) if other_lines_output_path else None
        )

        self.input_crs = input_crs
        self.output_crs = output_crs
        self.working_crs = working_crs

        self.snap_tolerance_m = float(snap_tolerance_m)
        self.prune_leaf_length_m = float(prune_leaf_length_m)
        self.prune_iterations = int(prune_iterations)

        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self.keep_intersection_buffer_m = float(keep_intersection_buffer_m)
        self.original_vertex_buffer_m = float(original_vertex_buffer_m)

        self.log = logger.bind(
            builder=self.__class__.__name__,
            input=str(self.input_path),
        )

    # ============================================================
    # Public API
    # ============================================================

    def build(self, save: bool = True) -> CoastlineBuildResult:
        gdf = self._read()

        if gdf.crs is None:
            self.log.warning(f"Input CRS missing, assuming {self.input_crs}")
            gdf = gdf.set_crs(self.input_crs)

        work_crs = self.working_crs or gdf.estimate_utm_crs()
        if work_crs is None:
            raise ValueError("Failed to estimate working UTM CRS")

        self.log.info(f"Working CRS: {work_crs}")
        gdf_work = gdf.to_crs(work_crs)

        raw_lines = self._extract_lines(gdf_work)
        original_points = self._collect_original_vertices(raw_lines)

        snapped_lines = self._snap_lines(raw_lines)
        noded_lines = self._node_lines(snapped_lines)

        graph = self._build_graph(noded_lines)
        self.log.info(
            f"Graph built: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}"
        )

        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        if not components:
            raise ValueError("No connected components found")

        main_component = max(components, key=lambda g: g.number_of_edges())
        other_components = [g for g in components if g is not main_component]

        self.log.info(
            f"Main component: nodes={main_component.number_of_nodes()}, "
            f"edges={main_component.number_of_edges()}"
        )

        tree = self._maximum_spanning_tree(main_component)
        trunk_tree = self._prune_short_leaves(tree)

        if trunk_tree.number_of_edges() == 0:
            raise ValueError("Trunk tree is empty after pruning")

        backbone_nodes = self._tree_diameter_path(trunk_tree)
        main_edge_keys = self._path_to_edge_keys(backbone_nodes)

        intersection_points = self._collect_intersection_points(main_component)

        result_work = self._build_result(
            main_component=main_component,
            main_edge_keys=main_edge_keys,
            other_components=other_components,
            original_points=original_points,
            intersection_points=intersection_points,
            crs=work_crs,
        )

        result = CoastlineBuildResult(
            coastline_gdf=result_work.coastline_gdf.to_crs(self.output_crs),
            other_lines_gdf=result_work.other_lines_gdf.to_crs(self.output_crs),
        )

        if save:
            self.save_result(result)

        self.log.success(
            f"Done: main={len(result.coastline_gdf)}, other={len(result.other_lines_gdf)}"
        )
        return result

    def save_result(self, result: CoastlineBuildResult) -> None:
        if self.coastline_output_path is not None:
            self.coastline_output_path.parent.mkdir(parents=True, exist_ok=True)
            result.coastline_gdf.to_file(self.coastline_output_path, driver="GeoJSON")
            self.log.info(f"Saved main coastline to {self.coastline_output_path}")

        if self.other_lines_output_path is not None:
            self.other_lines_output_path.parent.mkdir(parents=True, exist_ok=True)
            result.other_lines_gdf.to_file(self.other_lines_output_path, driver="GeoJSON")
            self.log.info(f"Saved other lines to {self.other_lines_output_path}")

    # ============================================================
    # Input
    # ============================================================

    def _read(self) -> gpd.GeoDataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        gdf = gpd.read_file(self.input_path)
        if gdf.empty:
            raise ValueError("Input GeoJSON is empty")

        self.log.info(f"Loaded {len(gdf)} feature(s)")
        return gdf

    def _extract_lines(self, gdf: gpd.GeoDataFrame) -> list[LineString]:
        lines: list[LineString] = []

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, LineString):
                if len(geom.coords) >= 2:
                    lines.append(geom)

            elif isinstance(geom, MultiLineString):
                for part in geom.geoms:
                    if len(part.coords) >= 2:
                        lines.append(part)

        if not lines:
            raise ValueError("No linear geometries found")

        self.log.info(f"Extracted {len(lines)} line(s)")
        return lines

    def _collect_original_vertices(self, lines: list[LineString]) -> list[Point]:
        points: list[Point] = []
        for line in lines:
            for xy in line.coords:
                points.append(Point(xy))
        self.log.info(f"Collected {len(points)} original vertex points")
        return points

    def _snap_lines(self, lines: list[LineString]) -> list[LineString]:
        if self.snap_tolerance_m <= 0:
            return lines

        self.log.info(f"Snapping with tolerance={self.snap_tolerance_m} m")

        snapped = lines[:]
        for i in range(len(snapped)):
            current = snapped[i]
            for j in range(len(snapped)):
                if i == j:
                    continue
                current = snap(current, snapped[j], self.snap_tolerance_m)
            snapped[i] = current

        return snapped

    def _node_lines(self, lines: list[LineString]) -> list[LineString]:
        self.log.info("Noding network")

        combined = unary_union(lines)
        noded = shapely_node(combined)

        if isinstance(noded, LineString):
            parts = [noded] if len(noded.coords) >= 2 else []
        elif isinstance(noded, MultiLineString):
            parts = [g for g in noded.geoms if len(g.coords) >= 2]
        else:
            raise TypeError(f"Unexpected geometry after noding: {noded.geom_type}")

        if not parts:
            raise ValueError("No lines after noding")

        self.log.info(f"Noded into {len(parts)} segment(s)")
        return parts

    # ============================================================
    # Graph
    # ============================================================

    def _ck(self, xy: tuple[float, float]) -> tuple[float, float]:
        return round(xy[0], self._COORD_ROUND), round(xy[1], self._COORD_ROUND)

    def _ek(self, u, v) -> tuple:
        return tuple(sorted((u, v)))

    def _build_graph(self, lines: list[LineString]) -> nx.Graph:
        g = nx.Graph()

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            start = self._ck(coords[0])
            end = self._ck(coords[-1])

            geom = LineString(coords)
            weight = float(geom.length)

            if g.has_edge(start, end):
                if g[start][end]["weight"] < weight:
                    g[start][end]["weight"] = weight
                    g[start][end]["geometry"] = geom
            else:
                g.add_edge(
                    start,
                    end,
                    geometry=geom,
                    weight=weight,
                    edge_key=self._ek(start, end),
                )

        if g.number_of_edges() == 0:
            raise ValueError("Graph has no edges")

        return g

    # ============================================================
    # Tree / trunk extraction
    # ============================================================

    def _maximum_spanning_tree(self, graph: nx.Graph) -> nx.Graph:
        self.log.info("Building maximum spanning tree")
        tree = nx.maximum_spanning_tree(graph, weight="weight")
        self.log.info(
            f"Maximum spanning tree: nodes={tree.number_of_nodes()}, "
            f"edges={tree.number_of_edges()}"
        )
        return tree

    def _leaf_branch_length(self, tree: nx.Graph, leaf) -> tuple[float, list]:
        if tree.degree(leaf) != 1:
            return 0.0, []

        branch_edges = []
        total_length = 0.0

        prev = None
        current = leaf

        while True:
            neighbors = [n for n in tree.neighbors(current) if n != prev]
            if not neighbors:
                break

            nxt = neighbors[0]
            edge_data = tree[current][nxt]
            branch_edges.append((current, nxt))
            total_length += float(edge_data["weight"])

            prev, current = current, nxt

            if tree.degree(current) != 2:
                break

        return total_length, branch_edges

    def _prune_short_leaves(self, tree: nx.Graph) -> nx.Graph:
        self.log.info(
            f"Pruning short leaves: threshold={self.prune_leaf_length_m} m, "
            f"iterations={self.prune_iterations}"
        )

        pruned = tree.copy()

        for iteration in range(self.prune_iterations):
            leaves = [n for n in pruned.nodes() if pruned.degree(n) == 1]
            to_remove = set()

            for leaf in leaves:
                branch_len, branch_edges = self._leaf_branch_length(pruned, leaf)
                if 0 < branch_len < self.prune_leaf_length_m:
                    for u, v in branch_edges:
                        to_remove.add(self._ek(u, v))

            if not to_remove:
                self.log.info(f"Pruning stopped at iteration {iteration + 1}")
                break

            for u, v in list(pruned.edges()):
                if self._ek(u, v) in to_remove:
                    pruned.remove_edge(u, v)

            isolated = [n for n in pruned.nodes() if pruned.degree(n) == 0]
            pruned.remove_nodes_from(isolated)

        self.log.info(
            f"Pruned tree: nodes={pruned.number_of_nodes()}, edges={pruned.number_of_edges()}"
        )
        return pruned

    def _farthest_node(self, tree: nx.Graph, source):
        lengths = nx.single_source_dijkstra_path_length(tree, source, weight="weight")
        return max(lengths, key=lengths.get)

    def _tree_diameter_path(self, tree: nx.Graph) -> list:
        self.log.info("Computing tree diameter path")

        start = next(iter(tree.nodes()))
        a = self._farthest_node(tree, start)
        b = self._farthest_node(tree, a)

        path = nx.dijkstra_path(tree, a, b, weight="weight")
        total_len = sum(tree[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))

        self.log.info(
            f"Tree diameter path: nodes={len(path)}, length={total_len:.2f} m"
        )
        return path

    def _path_to_edge_keys(self, path_nodes: list) -> set[tuple]:
        return {self._ek(u, v) for u, v in zip(path_nodes[:-1], path_nodes[1:])}

    def _collect_intersection_points(self, graph: nx.Graph) -> list[Point]:
        pts = []
        for node in graph.nodes():
            if graph.degree(node) >= 3:
                pts.append(Point(node))
        self.log.info(f"Collected {len(pts)} intersection point(s)")
        return pts

    # ============================================================
    # Geometry simplification based on original vertices
    # ============================================================

    @staticmethod
    def _segment_bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        diff = math.degrees(abs(a - b)) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        if diff > 90.0:
            diff = 180.0 - diff
        return diff

    def _is_protected_point(
        self,
        pt: Point,
        original_points_union,
        intersection_points_union,
    ) -> bool:
        if original_points_union is not None:
            if pt.distance(original_points_union) <= self.original_vertex_buffer_m:
                return True

        if intersection_points_union is not None:
            if pt.distance(intersection_points_union) <= self.keep_intersection_buffer_m:
                return True

        return False

    def _simplify_linestring_by_anchors(
        self,
        line: LineString,
        original_points_union,
        intersection_points_union,
    ) -> LineString:
        coords = list(line.coords)
        if len(coords) <= 2:
            return line

        kept = [coords[0]]

        for i in range(1, len(coords) - 1):
            prev_pt = coords[i - 1]
            cur_pt = coords[i]
            next_pt = coords[i + 1]

            p = Point(cur_pt)

            # Сохраняем все исходные вершины и точки пересечений
            if self._is_protected_point(
                p,
                original_points_union=original_points_union,
                intersection_points_union=intersection_points_union,
            ):
                kept.append(cur_pt)
                continue

            a1 = self._segment_bearing(prev_pt, cur_pt)
            a2 = self._segment_bearing(cur_pt, next_pt)
            diff = self._angle_diff_deg(a1, a2)

            # Если это реальный излом — тоже сохраняем
            if diff > self.angle_tolerance_deg:
                kept.append(cur_pt)

        kept.append(coords[-1])

        # дополнительная очистка дублей
        cleaned = [kept[0]]
        for pt in kept[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)

        if len(cleaned) < 2:
            return line

        return LineString(cleaned)

    def _simplify_main_geometries(
        self,
        main_geometries: list[LineString],
        original_points: list[Point],
        intersection_points: list[Point],
    ) -> list[LineString]:
        self.log.info(
            f"Simplifying main coastline using original vertices and intersections"
        )

        merged = linemerge(main_geometries)

        if isinstance(merged, LineString):
            lines = [merged]
        elif isinstance(merged, MultiLineString):
            lines = list(merged.geoms)
        else:
            lines = main_geometries

        original_union = unary_union(original_points) if original_points else None
        intersection_union = unary_union(intersection_points) if intersection_points else None

        simplified: list[LineString] = []
        for line in lines:
            s = self._simplify_linestring_by_anchors(
                line=line,
                original_points_union=original_union,
                intersection_points_union=intersection_union,
            )
            if s is not None and not s.is_empty and len(s.coords) >= 2:
                simplified.append(s)

        self.log.info(
            f"Simplified main geometries: {len(main_geometries)} -> {len(simplified)}"
        )
        return simplified

    # ============================================================
    # Output
    # ============================================================

    def _build_result(
        self,
        main_component: nx.Graph,
        main_edge_keys: set[tuple],
        other_components: list[nx.Graph],
        original_points: list[Point],
        intersection_points: list[Point],
        crs,
    ) -> CoastlineBuildResult:
        main_geometries: list[LineString] = []
        other_geometries: list[LineString] = []

        for u, v, data in main_component.edges(data=True):
            ek = self._ek(u, v)

            if ek in main_edge_keys:
                main_geometries.append(data["geometry"])
            else:
                other_geometries.append(data["geometry"])

        for component in other_components:
            for _, _, data in component.edges(data=True):
                other_geometries.append(data["geometry"])

        if not main_geometries:
            raise ValueError("No geometries selected for main coastline")

        simplified_main = self._simplify_main_geometries(
            main_geometries=main_geometries,
            original_points=original_points,
            intersection_points=intersection_points,
        )

        if not simplified_main:
            raise ValueError("Main coastline simplification produced no geometry")

        coastline_geom = (
            simplified_main[0]
            if len(simplified_main) == 1
            else MultiLineString(simplified_main)
        )

        coastline_gdf = gpd.GeoDataFrame(
            [
                {
                    "role": "main_coastline",
                    "geometry_type": coastline_geom.geom_type,
                    "segments_count": len(simplified_main),
                    "length_m": float(sum(g.length for g in simplified_main)),
                    "snap_tolerance_m": self.snap_tolerance_m,
                    "prune_leaf_length_m": self.prune_leaf_length_m,
                    "angle_tolerance_deg": self.angle_tolerance_deg,
                    "geometry": coastline_geom,
                }
            ],
            geometry="geometry",
            crs=crs,
        )

        other_lines_gdf = gpd.GeoDataFrame(
            [
                {
                    "role": "other_line",
                    "length_m": float(geom.length),
                    "snap_tolerance_m": self.snap_tolerance_m,
                    "prune_leaf_length_m": self.prune_leaf_length_m,
                    "geometry": geom,
                }
                for geom in other_geometries
                if geom is not None and not geom.is_empty
            ],
            geometry="geometry",
            crs=crs,
        )

        return CoastlineBuildResult(
            coastline_gdf=coastline_gdf,
            other_lines_gdf=other_lines_gdf,
        )


if __name__ == "__main__":
    from loguru import logger

    builder = MainCoastlineBuilder(
        # input_path="../../data/NovorossCoastlineAdded.geojson",
        input_path="../../../data/BlackSeaCoastlineS2Coast2023.geojson",
        # coastline_output_path="../../data/main_coastline.geojson",
        coastline_output_path="../../../data/black_sea_main_coastline.geojson",
        # other_lines_output_path="../../data/other_lines.geojson",
        other_lines_output_path="../../../data/black_sea_other_lines.geojson",
        input_crs="EPSG:4326",
        output_crs="EPSG:4326",
        working_crs=None,
        snap_tolerance_m=3.0,
        prune_leaf_length_m=80.0,
        prune_iterations=30,
        angle_tolerance_deg=1.0,
        keep_intersection_buffer_m=0.05,
        original_vertex_buffer_m=0.05,
    )

    result = builder.build(save=True)
    print(result)