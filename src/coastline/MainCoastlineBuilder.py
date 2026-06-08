from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
from loguru import logger
from shapely import node as shapely_node, segmentize, snap, unary_union
from shapely.geometry import LineString, MultiLineString


@dataclass
class CoastlineBuildResult:
    coastline_gdf: gpd.GeoDataFrame
    other_lines_gdf: gpd.GeoDataFrame


class MainCoastlineBuilder:
    """
    Надёжное выделение главной трассы береговой линии.

    Ключевая идея:
    - работаем в метрической CRS;
    - строим плотный граф по всем вершинам;
    - берём крупнейшую компоненту;
    - превращаем её в дерево;
    - prune короткие боковые ветки;
    - берём диаметр оставшегося дерева;
    - сохраняем главную трассу как набор сегментов, а не forcing в один LineString.

    Это специально сделано, чтобы не терять сегменты на развилках, где line_merge
    не может сшить линии через узлы степени 3+.
    """

    _COORD_ROUND = 2  # сантиметровая точность в метрах

    def __init__(
        self,
        input_path: str | Path,
        coastline_output_path: str | Path | None = None,
        other_lines_output_path: str | Path | None = None,
        input_crs: str = "EPSG:4326",
        output_crs: str = "EPSG:4326",
        working_crs: str | None = None,
        snap_tolerance_m: float = 3.0,
        segmentize_step_m: float = 10.0,
        prune_leaf_length_m: float = 50.0,
        prune_iterations: int = 20,
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
        self.segmentize_step_m = float(segmentize_step_m)
        self.prune_leaf_length_m = float(prune_leaf_length_m)
        self.prune_iterations = int(prune_iterations)

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

        lines = self._extract_lines(gdf_work)
        lines = self._snap_lines(lines)
        lines = self._segmentize_lines(lines)
        lines = self._node_lines(lines)

        graph = self._build_graph(lines)
        self.log.info(
            f"Graph built: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}"
        )

        components = [
            graph.subgraph(c).copy()
            for c in nx.connected_components(graph)
        ]
        if not components:
            raise ValueError("No connected components found")

        main_component = max(components, key=lambda g: g.number_of_edges())
        other_components = [g for g in components if g is not main_component]

        self.log.info(
            f"Main component: nodes={main_component.number_of_nodes()}, "
            f"edges={main_component.number_of_edges()}"
        )

        # 1) Удаляем циклы: получаем дерево
        tree = self._maximum_spanning_tree(main_component)

        # 2) Подрезаем короткие листья
        trunk_tree = self._prune_short_leaves(tree)

        if trunk_tree.number_of_edges() == 0:
            raise ValueError("Trunk tree is empty after pruning")

        # 3) Диаметр дерева = главная трасса
        backbone_nodes = self._tree_diameter_path(trunk_tree)
        main_edge_keys = self._path_to_edge_keys(backbone_nodes)

        # 4) Собираем результат
        result_work = self._build_result(
            main_component=main_component,
            trunk_tree=trunk_tree,
            main_edge_keys=main_edge_keys,
            other_components=other_components,
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
    # Input and preprocessing
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

    def _segmentize_lines(self, lines: list[LineString]) -> list[LineString]:
        if self.segmentize_step_m <= 0:
            return lines

        self.log.info(f"Segmentizing with step={self.segmentize_step_m} m")
        return [
            segmentize(line, max_segment_length=self.segmentize_step_m)
            for line in lines
        ]

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
    # Graph building
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

            for a, b in zip(coords[:-1], coords[1:]):
                u = self._ck(a)
                v = self._ck(b)
                if u == v:
                    continue

                geom = LineString([a, b])
                weight = float(geom.length)

                if g.has_edge(u, v):
                    # сохраняем более длинную геометрию как representative
                    if g[u][v]["weight"] < weight:
                        g[u][v]["weight"] = weight
                        g[u][v]["geometry"] = geom
                else:
                    g.add_edge(
                        u,
                        v,
                        geometry=geom,
                        weight=weight,
                        edge_key=self._ek(u, v),
                    )

        if g.number_of_edges() == 0:
            raise ValueError("Graph has no edges")

        return g

    # ============================================================
    # Tree / trunk extraction
    # ============================================================

    def _maximum_spanning_tree(self, graph: nx.Graph) -> nx.Graph:
        """
        Используем maximum spanning tree, чтобы сохранить длинные магистральные рёбра,
        а не случайно выбросить их как в minimum spanning tree.
        """
        self.log.info("Building maximum spanning tree")
        tree = nx.maximum_spanning_tree(graph, weight="weight")
        self.log.info(
            f"Maximum spanning tree: nodes={tree.number_of_nodes()}, "
            f"edges={tree.number_of_edges()}"
        )
        return tree

    def _leaf_branch_length(self, tree: nx.Graph, leaf) -> tuple[float, list]:
        """
        Идём от leaf до первого узла со степенью != 2.
        Возвращаем длину ветви и список рёбер этой ветви.
        """
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
        """
        Многократно удаляем короткие листовые ветки.
        Это намного стабильнее, чем пытаться угадать магистраль одним shortest path.
        """
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
        """
        Для дерева double sweep даёт диаметр корректно.
        """
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
        return {
            self._ek(u, v)
            for u, v in zip(path_nodes[:-1], path_nodes[1:])
        }

    # ============================================================
    # Output
    # ============================================================

    def _build_result(
        self,
        main_component: nx.Graph,
        trunk_tree: nx.Graph,
        main_edge_keys: set[tuple],
        other_components: list[nx.Graph],
        crs,
    ) -> CoastlineBuildResult:
        """
        Главное отличие:
        НЕ склеиваем main в один LineString через line_merge(),
        потому что на узлах степени 3+ часть трассы визуально теряется.
        Сохраняем main как набор сегментов.
        """
        main_geometries: list[LineString] = []
        other_geometries: list[LineString] = []

        trunk_edge_keys = {
            self._ek(u, v)
            for u, v in trunk_tree.edges()
        }

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

        main_multiline = MultiLineString(main_geometries)

        coastline_gdf = gpd.GeoDataFrame(
            [
                {
                    "role": "main_coastline",
                    "geometry_type": "MultiLineString",
                    "segments_count": len(main_geometries),
                    "length_m": float(sum(g.length for g in main_geometries)),
                    "snap_tolerance_m": self.snap_tolerance_m,
                    "segmentize_step_m": self.segmentize_step_m,
                    "prune_leaf_length_m": self.prune_leaf_length_m,
                    "geometry": main_multiline,
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
                    "segmentize_step_m": self.segmentize_step_m,
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
        input_path="../../data/NovorossCoastlineVectorS2Coast2023.geojson",
        coastline_output_path="../../output/main_coastline.geojson",
        other_lines_output_path="../../output/other_lines.geojson",
        input_crs="EPSG:4326",
        output_crs="EPSG:4326",
        working_crs=None,
        snap_tolerance_m=3.0,
        segmentize_step_m=10.0,
        prune_leaf_length_m=80.0,
        prune_iterations=30,
    )

    result = builder.build(save=True)
    print(result)