from __future__ import annotations

from urllib.parse import urlencode

import requests

from .models import GridPoint


class OpenMeteoArchiveClient:
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, user_agent: str = "BreachTheBeach/0.1.0", timeout: float = 120.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

    def build_params(
        self,
        points: list[GridPoint],
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
        model: str,
        timezone: str,
        cell_selection: str,
    ) -> dict[str, str]:
        return {
            "latitude": ",".join(f"{point.lat:.3f}" for point in points),
            "longitude": ",".join(f"{point.lon:.3f}" for point in points),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(daily_variables),
            "timezone": timezone,
            "models": model,
            "cell_selection": cell_selection,
        }

    def fetch(
        self,
        points: list[GridPoint],
        start_date: str,
        end_date: str,
        daily_variables: tuple[str, ...],
        model: str,
        timezone: str,
        cell_selection: str,
    ) -> tuple[dict | list[dict], str]:
        params = self.build_params(
            points=points,
            start_date=start_date,
            end_date=end_date,
            daily_variables=daily_variables,
            model=model,
            timezone=timezone,
            cell_selection=cell_selection,
        )

        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()

        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(payload.get("reason", "Open-Meteo returned an error"))

        source_url = f"{self.BASE_URL}?{urlencode(params)}"
        return payload, source_url
