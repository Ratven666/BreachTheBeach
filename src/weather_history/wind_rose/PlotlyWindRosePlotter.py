from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.weather_history.wind_rose.WindRose import WindRose


class PlotlyWindRosePlotter:
    def build_barpolar(
        self,
        wind_rose: WindRose,
        title: str | None = None,
    ) -> go.Figure:
        table = wind_rose.table_data
        fig = go.Figure()

        for i in range(table.speed_class_count):
            label = f"{table.speed_bins[i]:.2f}–{table.speed_bins[i+1]:.2f}"
            if wind_rose.ws_unit:
                label += f" {wind_rose.ws_unit}"

            fig.add_trace(
                go.Barpolar(
                    r=table.frequencies_percent[i].tolist(),
                    theta=table.direction_labels,
                    name=label,
                )
            )

        fig.update_layout(
            title=title or wind_rose.title or "Wind Rose",
            polar=dict(
                radialaxis=dict(ticksuffix="%"),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            legend=dict(title="Wind speed bins"),
            template="plotly_white",
        )
        return fig

    def save_html(
        self,
        wind_rose: WindRose,
        output_path: str | Path,
        title: str | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = self.build_barpolar(wind_rose=wind_rose, title=title)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    def to_table_frame(self, wind_rose: WindRose) -> pd.DataFrame:
        return wind_rose.table_data.as_dataframe()
