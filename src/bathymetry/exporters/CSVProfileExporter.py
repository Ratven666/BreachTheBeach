from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.bathymetry.domain.models import BathymetryProfile
from src.bathymetry.exporters.base import ProfileExportStrategy


class CSVProfileExporter(ProfileExportStrategy):
    def export(self, profile: BathymetryProfile, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "distance_m": profile.distances,
                "depth_m": profile.depths,
                "lat": [p.lat for p in profile.points],
                "lon": [p.lon for p in profile.points],
            }
        )
        df.to_csv(path, index=False, float_format="%.6f")
        logger.info(f"Profile exported to CSV: {path}")
        return path
