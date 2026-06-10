from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from .WindFetchCalculator import WindFetchCalculator
from .WindFetchResult import WindFetchResult


class WindFetchVisualizer:
    def __init__(self, calculator: WindFetchCalculator) -> None:
        self.calculator = calculator

    def plot(
        self,
        results: list[WindFetchResult],
        output_path: str | Path,
        figsize: tuple[float, float] = (12, 12),
    ) -> Path:
        coastline = self.calculator.coastline_gdf
        points = self.calculator.to_geodataframe(results)
        rays = self.calculator.to_rays_geodataframe(results)

        fig, ax = plt.subplots(figsize=figsize)

        coastline.plot(ax=ax, color="black", linewidth=1.0)
        rays.plot(
            ax=ax,
            column="fetch_length_m",
            cmap="viridis",
            linewidth=1.0,
            legend=True,
        )
        points.plot(ax=ax, color="red", markersize=8)

        ax.set_title("Wind Fetch Length")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        logger.success(f"Saved wind fetch plot: {output_path}")
        return output_path