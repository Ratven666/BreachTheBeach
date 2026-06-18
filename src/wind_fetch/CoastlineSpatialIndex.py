from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree

from .geometry_utils import iter_lines


class CoastlineSpatialIndex:
    def __init__(self, combined_gdf: gpd.GeoDataFrame) -> None:
        self.gdf = combined_gdf.reset_index(drop=True).copy()

        line_geoms = []
        line_meta = []

        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            for part in iter_lines(geom):
                line_geoms.append(part)
                line_meta.append(
                    {
                        "source_index": idx,
                        "coastline_role": row.get("coastline_role"),
                    }
                )

        self.line_geoms = line_geoms
        self.line_meta = pd.DataFrame(line_meta)
        self.tree = STRtree(self.line_geoms) if self.line_geoms else None

    def query(self, geom):
        if self.tree is None:
            return []
        indices = self.tree.query(geom)
        return [self.line_geoms[int(i)] for i in indices]
