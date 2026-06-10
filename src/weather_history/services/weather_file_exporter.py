from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd


def slugify(value: Any) -> str:
    text = str(value).strip()
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "unknown"


def export_gdf(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
    driver: str = "GeoJSON",
    layer_name: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gpkg" or driver.upper() == "GPKG":
        gdf.to_file(output_path, driver="GPKG", layer=layer_name or "layer", index=False)
    else:
        gdf.to_file(output_path, driver=driver, index=False)

    return output_path


def write_manifest(output_dir: str | Path, exported_files: list[Path]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "files_count": len(exported_files),
        "files": [str(path.relative_to(output_dir)) for path in exported_files],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path
