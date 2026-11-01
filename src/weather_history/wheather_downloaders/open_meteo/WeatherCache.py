from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .models import CacheSegment, GridPoint


class WeatherCache:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_variables_key(daily_variables: tuple[str, ...]) -> str:
        payload = "|".join(sorted(daily_variables)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def point_dir(self, point: GridPoint, model: str) -> Path:
        return self.root_dir / model / f"lat_{point.lat_label}" / f"lon_{point.lon_label}"

    def segment_paths(
        self,
        point: GridPoint,
        model: str,
        start_date: str,
        end_date: str,
        variables_key: str,
    ) -> CacheSegment:
        directory = self.point_dir(point, model)
        directory.mkdir(parents=True, exist_ok=True)

        stem = f"{start_date}_{end_date}_{variables_key}"
        return CacheSegment(
            point=point,
            start_date=start_date,
            end_date=end_date,
            variables_key=variables_key,
            json_path=directory / f"{stem}.json",
            metadata_path=directory / f"{stem}.meta.json",
        )

    def has_segment(
        self,
        point: GridPoint,
        model: str,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
    ) -> bool:
        variables_key = self.build_variables_key(daily_variables)
        segment = self.segment_paths(
            point=point,
            model=model,
            start_date=start_date,
            end_date=end_date,
            variables_key=variables_key,
        )
        return segment.json_path.exists() and segment.metadata_path.exists()

    def load_segment(
        self,
        point: GridPoint,
        model: str,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
    ) -> dict:
        variables_key = self.build_variables_key(daily_variables)
        segment = self.segment_paths(
            point=point,
            model=model,
            start_date=start_date,
            end_date=end_date,
            variables_key=variables_key,
        )
        return json.loads(segment.json_path.read_text(encoding="utf-8"))

    def load_segment_by_key(self, cache_key: str) -> dict:
        json_path = Path(cache_key)
        return json.loads(json_path.read_text(encoding="utf-8"))

    def iter_segments_metadata(
        self,
        point: GridPoint,
        model: str,
        daily_variables: tuple[str, ...],
        timezone: str,
        cell_selection: str,
    ) -> list[dict]:
        variables_key = self.build_variables_key(daily_variables)
        directory = self.point_dir(point, model)

        if not directory.exists():
            return []

        items: list[dict] = []

        for metadata_path in sorted(directory.glob("*.meta.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if metadata.get("model") != model:
                continue
            if metadata.get("variables_key") != variables_key:
                continue
            if metadata.get("timezone") != timezone:
                continue
            if metadata.get("cell_selection") != cell_selection:
                continue

            start_date = metadata.get("start_date")
            end_date = metadata.get("end_date")
            if not start_date or not end_date:
                continue

            json_path = metadata_path.with_name(metadata_path.name.replace(".meta.json", ".json"))
            if not json_path.exists():
                continue

            items.append(
                {
                    "cache_key": str(json_path),
                    "metadata_key": str(metadata_path),
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily_variables": metadata.get("daily_variables", []),
                    "timezone": metadata.get("timezone"),
                    "cell_selection": metadata.get("cell_selection"),
                }
            )

        return items

    def save_segment(
        self,
        point: GridPoint,
        model: str,
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
        payload: dict,
        source_url: str,
        timezone: str,
        cell_selection: str,
    ) -> CacheSegment:
        variables_key = self.build_variables_key(daily_variables)
        segment = self.segment_paths(
            point=point,
            model=model,
            start_date=start_date,
            end_date=end_date,
            variables_key=variables_key,
        )

        segment.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metadata = {
            "requested_point": asdict(point),
            "resolved_point": {
                "lat": payload.get("latitude"),
                "lon": payload.get("longitude"),
            },
            "model": model,
            "start_date": start_date,
            "end_date": end_date,
            "daily_variables": list(daily_variables),
            "variables_key": variables_key,
            "timezone": timezone,
            "cell_selection": cell_selection,
            "source_url": source_url,
            "downloaded_at": datetime.now(UTC).isoformat(),
        }

        segment.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return segment
