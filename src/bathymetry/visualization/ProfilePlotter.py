from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.bathymetry.domain.models import BathymetryProfile


class ProfilePlotter:
    def __init__(self, figsize: tuple[int, int] = (12, 4)) -> None:
        self._figsize = figsize

    def plot(
        self,
        profile: BathymetryProfile,
        title: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        dist_km = profile.distances / 1000.0

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.plot(dist_km, profile.depths, color="black", linewidth=1.2)

        zero = np.zeros_like(profile.depths)
        ax.fill_between(dist_km, profile.depths, zero, where=(profile.depths < 0), alpha=0.4, color="#4393c3")
        ax.fill_between(dist_km, profile.depths, zero, where=(profile.depths >= 0), alpha=0.4, color="#a1d99b")

        ax.axhline(0, color="navy", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Elevation / Depth (m)")
        ax.set_title(title or "Bathymetry profile")
        ax.grid(True, alpha=0.3)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig
