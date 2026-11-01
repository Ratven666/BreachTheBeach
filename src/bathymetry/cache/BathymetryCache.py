from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.base.BBox import BBox
from src.bathymetry.domain.models import BathymetryGrid


class BathymetryCache:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger.bind(cls=self.__class__.__name__)

    def has(self, bbox: BBox, source: str) -> bool:
        key = self._cache_key(bbox, source)
        return self._npz_path(key).exists() and self._meta_path(key).exists()

    def load(self, bbox: BBox, source: str) -> BathymetryGrid:
        key = self._cache_key(bbox, source)
        self._log.debug(f"Cache HIT: key={key[:12]}... source={source}")

        arrays = np.load(self._npz_path(key))
        meta = json.loads(self._meta_path(key).read_text(encoding="utf-8"))

        return BathymetryGrid(
            lats=arrays["lats"],
            lons=arrays["lons"],
            z=arrays["z"],
            source=meta["source"],
            resolution_arcsec=meta.get("resolution_arcsec", 15.0),
        )

    def save(self, bbox: BBox, grid: BathymetryGrid) -> None:
        key = self._cache_key(bbox, grid.source)
        self._log.debug(f"Cache SAVE: key={key[:12]}... source={grid.source}")

        np.savez_compressed(
            self._npz_path(key),
            lats=grid.lats,
            lons=grid.lons,
            z=grid.z,
        )

        meta = {
            "bbox": {
                "south": bbox.south,
                "west": bbox.west,
                "north": bbox.north,
                "east": bbox.east,
            },
            "source": grid.source,
            "shape": list(grid.shape),
            "resolution_arcsec": grid.resolution_arcsec,
            "depth_range": [grid.min_depth, grid.max_depth],
            "cached_at": datetime.now(UTC).isoformat(),
        }

        self._meta_path(key).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def invalidate(self, bbox: BBox, source: str) -> None:
        key = self._cache_key(bbox, source)
        for path in (self._npz_path(key), self._meta_path(key)):
            if path.exists():
                path.unlink()
        self._log.info(f"Cache invalidated: key={key[:12]}...")

    @staticmethod
    def _cache_key(bbox: BBox, source: str) -> str:
        payload = f"{bbox.south}|{bbox.west}|{bbox.north}|{bbox.east}|{source}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _npz_path(self, key: str) -> Path:
        return self.root_dir / f"{key}.npz"

    def _meta_path(self, key: str) -> Path:
        return self.root_dir / f"{key}.meta.json"
