from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.weather_history.wind_rose.WindRose import WindRose


class MatplotlibWindRosePlotter:
    def plot_bar(
        self,
        wind_rose: WindRose,
        figsize: tuple[float, float] = (8, 8),
        cmap: str = "viridis",
        opening: float = 0.85,
        title: str | None = None,
    ):
        table = wind_rose.table_data
        theta = np.deg2rad(table.direction_centers)
        width = np.deg2rad(360.0 / table.sector_count) * float(opening)

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.95, table.speed_class_count))

        bottom = np.zeros(table.sector_count, dtype=float)

        for i in range(table.speed_class_count):
            values = table.frequencies_percent[i]
            label = f"{table.speed_bins[i]:.2f}–{table.speed_bins[i+1]:.2f}"
            if wind_rose.ws_unit:
                label += f" {wind_rose.ws_unit}"

            ax.bar(
                theta,
                values,
                width=width,
                bottom=bottom,
                color=colors[i],
                edgecolor="white",
                linewidth=0.7,
                align="center",
                label=label,
            )
            bottom += values

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(theta)
        ax.set_xticklabels(table.direction_labels)
        ax.set_title(title or wind_rose.title or "Wind Rose")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
        return fig, ax

    def save_bar(
        self,
        wind_rose: WindRose,
        output_path: str | Path,
        figsize: tuple[float, float] = (8, 8),
        cmap: str = "viridis",
        opening: float = 0.85,
        title: str | None = None,
        dpi: int = 180,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, _ = self.plot_bar(
            wind_rose=wind_rose,
            figsize=figsize,
            cmap=cmap,
            opening=opening,
            title=title,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path
