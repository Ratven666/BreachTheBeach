from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.bathymetry.domain.models import BathymetryGrid


class BathymetryGridPlotter:
    def __init__(self, figsize: tuple[int, int] = (10, 8)) -> None:
        self._figsize = figsize
        self._cmap = self._build_topobathy_cmap()

    def plot(
        self,
        grid: BathymetryGrid,
        title: Optional[str] = None,
        contour_interval: Optional[float] = None,
        output_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        lons, lats = np.meshgrid(grid.lons, grid.lats)

        z = np.asarray(grid.z, dtype=float)
        norm = self._build_norm(z)

        fig, ax = plt.subplots(figsize=self._figsize)
        mesh = ax.pcolormesh(
            lons,
            lats,
            z,
            cmap=self._cmap,
            norm=norm,
            shading="auto",
        )

        cbar = fig.colorbar(mesh, ax=ax, label="Elevation / Depth (m)")
        cbar.ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)

        if contour_interval is not None:
            finite = z[np.isfinite(z)]
            if finite.size > 0:
                vmin = np.floor(finite.min() / contour_interval) * contour_interval
                vmax = np.ceil(finite.max() / contour_interval) * contour_interval
                levels = np.arange(vmin, vmax + contour_interval, contour_interval)

                if len(levels) > 0:
                    ax.contour(
                        lons,
                        lats,
                        z,
                        levels=levels,
                        colors="black",
                        linewidths=0.4,
                        alpha=0.35,
                    )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title or f"Bathymetry — {grid.source}")
        ax.set_aspect("equal")

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    @staticmethod
    def _build_norm(z: np.ndarray):
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            return mcolors.Normalize(vmin=-1.0, vmax=1.0)

        vmin = float(finite.min())
        vmax = float(finite.max())

        has_sea = vmin < 0
        has_land = vmax > 0

        if has_sea and has_land:
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def _build_topobathy_cmap() -> LinearSegmentedColormap:
        sea = plt.cm.Blues_r(np.linspace(0.15, 0.95, 256))
        land = plt.cm.terrain(np.linspace(0.25, 1.00, 256))
        colors = np.vstack((sea, land))
        return LinearSegmentedColormap.from_list("topobathy_zero_centered", colors)
