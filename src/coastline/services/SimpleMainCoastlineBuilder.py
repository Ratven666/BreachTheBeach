from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString


@dataclass
class CoastlineBuildResult:
    coastline_gdf: gpd.GeoDataFrame
    other_lines_gdf: gpd.GeoDataFrame


class SimpleCachedCoastlineBuilder:
    """
    Упрощённый и быстрый сборщик береговой линии.

    Логика:
    - берём исходные линии как есть;
    - приводим их в метрическую CRS;
    - собираем локальные цепочки по близости крайних точек;
    - затем глобально сшиваем между собой все цепочки;
    - после этого выбираем самую длинную цепочку как main coastline;
    - остальные цепочки сохраняем как other lines.
    """

    def __init__(
        self,
        input_path: str | Path,
        coastline_output_path: str | Path | None = None,
        other_lines_output_path: str | Path | None = None,
        input_crs: str = "EPSG:4326",
        output_crs: str = "EPSG:4326",
        working_crs: str | None = None,
        max_endpoint_distance_m: float = 150.0,
        local_max_passes: int = 10,
        global_max_passes: int = 10,
    ) -> None:
        self.input_path = Path(input_path)
        self.coastline_output_path = Path(coastline_output_path) if coastline_output_path else None
        self.other_lines_output_path = Path(other_lines_output_path) if other_lines_output_path else None

        self.input_crs = input_crs
        self.output_crs = output_crs
        self.working_crs = working_crs

        self.max_endpoint_distance_m = float(max_endpoint_distance_m)
        self.local_max_passes = int(local_max_passes)
        self.global_max_passes = int(global_max_passes)

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
            raise ValueError("Failed to estimate working CRS")

        self.log.info(f"Working CRS: {work_crs}")
        gdf_work = gdf.to_crs(work_crs)

        raw_lines = self._extract_lines(gdf_work)
        self.log.info(f"Loaded {len(raw_lines)} raw line(s)")

        chains = self._build_all_chains(
            lines=raw_lines,
            max_dist_m=self.max_endpoint_distance_m,
            max_passes=self.local_max_passes,
        )

        self.log.info(f"Local chains built: {len(chains)}")

        chains = self._global_merge_cycle(
            chains=chains,
            max_dist_m=self.max_endpoint_distance_m,
            max_passes=self.global_max_passes,
        )

        self.log.info(f"Global chains built: {len(chains)}")

        if not chains:
            raise ValueError("No chains produced")

        main_line = max(chains, key=lambda g: g.length)
        other_lines = [g for g in chains if g is not main_line]

        result_work = CoastlineBuildResult(
            coastline_gdf=gpd.GeoDataFrame(
                [{
                    "role": "main_coastline",
                    "segments_count": 1,
                    "length_m": float(main_line.length),
                    "max_endpoint_distance_m": self.max_endpoint_distance_m,
                    "geometry": main_line,
                }],
                geometry="geometry",
                crs=work_crs,
            ),
            other_lines_gdf=gpd.GeoDataFrame(
                [
                    {
                        "role": "other_line",
                        "length_m": float(line.length),
                        "max_endpoint_distance_m": self.max_endpoint_distance_m,
                        "geometry": line,
                    }
                    for line in other_lines
                    if line is not None and not line.is_empty
                ],
                geometry="geometry",
                crs=work_crs,
            ),
        )

        result = CoastlineBuildResult(
            coastline_gdf=result_work.coastline_gdf.to_crs(self.output_crs),
            other_lines_gdf=result_work.other_lines_gdf.to_crs(self.output_crs),
        )

        if save:
            self.save_result(result)

        self.log.success(
            f"Done: main length={main_line.length:.2f} m, other lines={len(other_lines)}"
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

        return lines

    # ============================================================
    # Geometry helpers
    # ============================================================

    @staticmethod
    def _line_endpoints(line: LineString):
        coords = list(line.coords)
        return coords[0], coords[-1]

    @staticmethod
    def _reverse_line(line: LineString) -> LineString:
        return LineString(list(line.coords)[::-1])

    @staticmethod
    def _distance(p1, p2) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    def _best_join_mode(self, a: LineString, b: LineString):
        a0, a1 = self._line_endpoints(a)
        b0, b1 = self._line_endpoints(b)

        variants = [
            ("append_forward", self._distance(a1, b0)),
            ("append_reverse", self._distance(a1, b1)),
            ("prepend_forward", self._distance(a0, b1)),
            ("prepend_reverse", self._distance(a0, b0)),
        ]
        return min(variants, key=lambda x: x[1])

    def _merge_pair(self, a: LineString, b: LineString, mode: str) -> LineString:
        if mode == "append_forward":
            left = list(a.coords)
            right = list(b.coords)
        elif mode == "append_reverse":
            left = list(a.coords)
            right = list(self._reverse_line(b).coords)
        elif mode == "prepend_forward":
            left = list(b.coords)
            right = list(a.coords)
        elif mode == "prepend_reverse":
            left = list(self._reverse_line(b).coords)
            right = list(a.coords)
        else:
            raise ValueError(f"Unknown merge mode: {mode}")

        if left[-1] == right[0]:
            coords = left + right[1:]
        else:
            coords = left + right

        return LineString(coords)

    # ============================================================
    # Local chain build
    # ============================================================

    def _build_all_chains(
        self,
        lines: list[LineString],
        max_dist_m: float,
        max_passes: int = 10,
    ) -> list[LineString]:
        chains = [
            line for line in lines
            if line is not None and not line.is_empty and len(line.coords) >= 2
        ]

        for pass_num in range(1, max_passes + 1):
            if len(chains) < 2:
                break

            used = set()
            new_chains: list[LineString] = []
            merges = 0

            self.log.info(f"Local pass {pass_num}: chains={len(chains)}")

            for i in range(len(chains)):
                if i in used:
                    continue

                a = chains[i]
                best_j = None
                best_mode = None
                best_gap = float("inf")

                for j in range(len(chains)):
                    if i == j or j in used:
                        continue

                    mode, gap = self._best_join_mode(a, chains[j])
                    if gap < best_gap:
                        best_j = j
                        best_mode = mode
                        best_gap = gap

                if best_j is not None and best_gap <= max_dist_m:
                    merged = self._merge_pair(a, chains[best_j], best_mode)
                    new_chains.append(merged)
                    used.add(i)
                    used.add(best_j)
                    merges += 1

                    self.log.info(
                        f"LOCAL attach {i} - {best_j}, mode={best_mode}, "
                        f"gap={best_gap:.2f} m, newlength={merged.length:.2f} m"
                    )
                else:
                    new_chains.append(a)
                    used.add(i)

            chains = new_chains

            self.log.info(
                f"Local pass {pass_num} done: merges={merges}, chains={len(chains)}"
            )

            if merges == 0:
                break

        return chains

    # ============================================================
    # Global chain merge
    # ============================================================

    def _global_merge_cycle(
        self,
        chains: list[LineString],
        max_dist_m: float,
        max_passes: int = 10,
    ) -> list[LineString]:
        chains = [
            c for c in chains
            if c is not None and not c.is_empty and len(c.coords) >= 2
        ]

        for pass_num in range(1, max_passes + 1):
            if len(chains) < 2:
                break

            used = set()
            new_chains: list[LineString] = []
            merges = 0
            max_len = max(line.length for line in chains) if chains else 0.0

            self.log.info(
                f"Global pass {pass_num}: chains={len(chains)}, max_length={max_len:.2f} m"
            )

            for i in range(len(chains)):
                if i in used:
                    continue

                a = chains[i]
                best_j = None
                best_mode = None
                best_gap = float("inf")

                for j in range(len(chains)):
                    if i == j or j in used:
                        continue

                    mode, gap = self._best_join_mode(a, chains[j])
                    if gap < best_gap:
                        best_j = j
                        best_mode = mode
                        best_gap = gap

                if best_j is not None and best_gap <= max_dist_m:
                    merged = self._merge_pair(a, chains[best_j], best_mode)
                    new_chains.append(merged)
                    used.add(i)
                    used.add(best_j)
                    merges += 1

                    self.log.info(
                        f"GLOBAL attach {i} - {best_j}, mode={best_mode}, "
                        f"gap={best_gap:.2f} m, newlength={merged.length:.2f} m"
                    )
                else:
                    new_chains.append(a)
                    used.add(i)

            chains = new_chains

            self.log.info(
                f"Global pass {pass_num} done: merges={merges}, chains={len(chains)}"
            )

            if merges == 0:
                self.log.info(f"Global cycle stopped at pass {pass_num}")
                break

        return chains


if __name__ == "__main__":
    builder = SimpleCachedCoastlineBuilder(
        input_path="data/NVRSK_BlackSeaCoastlineS2Coast2023.geojson",
        coastline_output_path="data/nvrsk_black_sea_main_coastline_fast.geojson",
        other_lines_output_path="data/nvrsk_black_sea_other_lines_fast.geojson",
        input_crs="EPSG:4326",
        output_crs="EPSG:4326",
        working_crs=None,
        max_endpoint_distance_m=150.0,
        local_max_passes=10,
        global_max_passes=10,
    )

    result = builder.build(save=True)
    print(result)