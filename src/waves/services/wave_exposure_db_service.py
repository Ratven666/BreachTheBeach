from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.waves.indices import WaveExposureRanker
from src.waves.storage import (
    WaveActivityRepository,
    WaveActivityRow,
    WaveActivitySummaryRow,
    WaveExposureIndexRow,
)


SECTOR_LABELS: tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


@dataclass(frozen=True, slots=True)
class WaveExposureBuildStats:
    point_count: int
    valid_index_count: int
    min_wer: float | None
    max_wer: float | None
    mean_wer: float | None


def _direction_to_sector(
    direction_deg: float,
) -> str:
    index = (
        int(
            (
                float(direction_deg)
                + 11.25
            )
            / 22.5
        )
        % 16
    )
    return SECTOR_LABELS[index]


def _top3_adjacent_sectors(
    sector_sums: dict[str, float],
) -> list[str]:
    ring = SECTOR_LABELS + SECTOR_LABELS

    best_index = 0
    best_sum = -1.0

    for index in range(len(SECTOR_LABELS)):
        current = sum(
            sector_sums.get(label, 0.0)
            for label in ring[index:index + 3]
        )

        if current > best_sum:
            best_sum = current
            best_index = index

    return list(
        ring[best_index:best_index + 3]
    )


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


class WaveExposureDatabaseBuilder:
    def __init__(
        self,
        repository: WaveActivityRepository,
        *,
        storm_percentile: float = 90.0,
        storm_threshold: float | None = None,
        dt_seconds: float = 86_400.0,
    ) -> None:
        if not (
            0.0
            < float(storm_percentile)
            < 100.0
        ):
            raise ValueError(
                "storm_percentile must be in (0, 100)"
            )

        if (
            storm_threshold is not None
            and float(storm_threshold) < 0.0
        ):
            raise ValueError(
                "storm_threshold must be >= 0 or None"
            )

        if float(dt_seconds) <= 0.0:
            raise ValueError(
                "dt_seconds must be > 0"
            )

        self._repository = repository
        self._storm_percentile = float(
            storm_percentile
        )
        self._storm_threshold = (
            None
            if storm_threshold is None
            else float(storm_threshold)
        )
        self._dt_seconds = float(dt_seconds)
        self._ranker = WaveExposureRanker()

    def run(self) -> WaveExposureBuildStats:
        self._repository.assert_schema_version()

        summaries = (
            self._repository.read_summaries()
        )
        activity_by_point = (
            self._repository
            .read_activity_by_point()
        )

        if not summaries:
            raise ValueError(
                "wave_activity_summary is empty. "
                "Run db_pypeline/"
                "5_1_wave_activity_to_db.py first."
            )

        records: list[dict] = []

        for index, summary in enumerate(
            summaries,
            start=1,
        ):
            activity = activity_by_point.get(
                summary.point_id,
                [],
            )

            record = self._compute_point(
                summary,
                activity,
            )
            records.append(record)

            if (
                index % 25 == 0
                or index == len(summaries)
            ):
                logger.info(
                    "WEI metrics calculated: {}/{}",
                    index,
                    len(summaries),
                )

        metrics_df = pd.DataFrame(records)

        logger.info(
            "Assigning R1-R4 quintile ranks "
            "and calculating WER"
        )

        ranked = self._ranker.rank(
            metrics_df
        )

        output_rows = [
            WaveExposureIndexRow(
                point_id=int(row["point_id"]),
                mean_cwef_wm=_optional_float(
                    row["mean_CWEF_Wm"]
                ),
                e_storm_mjm=_optional_float(
                    row["E_storm_MJm"]
                ),
                storm_threshold_wm=(
                    _optional_float(
                        row["storm_threshold_Wm"]
                    )
                ),
                storm_percentile=(
                    _optional_float(
                        row["storm_percentile"]
                    )
                ),
                k_dir=_optional_float(
                    row["K_dir"]
                ),
                cv=_optional_float(
                    row["CV"]
                ),
                n_days=int(row["n_days"]),
                n_storm_days=int(
                    row["n_storm_days"]
                ),
                top3_sectors=(
                    None
                    if pd.isna(
                        row["top3_sectors"]
                    )
                    or not row["top3_sectors"]
                    else str(
                        row["top3_sectors"]
                    )
                ),
                r1=_optional_int(row["R1"]),
                r2=_optional_int(row["R2"]),
                r3=_optional_int(row["R3"]),
                r4=_optional_int(row["R4"]),
                wer=_optional_float(row["WER"]),
            )
            for _, row in ranked.iterrows()
        ]

        written = (
            self._repository
            .replace_exposure_indices(
                output_rows
            )
        )
        self._repository.commit()
        self._repository.optimize()

        wer_values = (
            ranked["WER"]
            .dropna()
            .astype(float)
        )

        stats = WaveExposureBuildStats(
            point_count=written,
            valid_index_count=int(
                wer_values.size
            ),
            min_wer=(
                None
                if wer_values.empty
                else float(wer_values.min())
            ),
            max_wer=(
                None
                if wer_values.empty
                else float(wer_values.max())
            ),
            mean_wer=(
                None
                if wer_values.empty
                else float(wer_values.mean())
            ),
        )

        logger.success(
            "Wave exposure indices ready: "
            "points={}, valid_WER={}, "
            "min_WER={}, max_WER={}, mean_WER={}",
            stats.point_count,
            stats.valid_index_count,
            stats.min_wer,
            stats.max_wer,
            stats.mean_wer,
        )

        return stats

    def _compute_point(
        self,
        summary: WaveActivitySummaryRow,
        activity: list[WaveActivityRow],
    ) -> dict:
        n_days = int(summary.n_days)

        if n_days == 0:
            return {
                "point_id": summary.point_id,
                "mean_CWEF_Wm": None,
                "E_storm_MJm": None,
                "storm_threshold_Wm": None,
                "storm_percentile": None,
                "K_dir": None,
                "CV": None,
                "n_days": 0,
                "n_storm_days": 0,
                "top3_sectors": None,
            }

        if len(activity) > n_days:
            raise ValueError(
                "Active-row count exceeds total day count: "
                f"point_id={summary.point_id}, "
                f"active_rows={len(activity)}, "
                f"n_days={n_days}"
            )

        active_values = np.asarray(
            [
                float(row.cwef_wm)
                for row in activity
            ],
            dtype=np.float64,
        )

        if active_values.size:
            if not np.all(
                np.isfinite(active_values)
            ):
                raise ValueError(
                    "Non-finite active CWEF values: "
                    f"point_id={summary.point_id}"
                )

            if np.any(active_values <= 0.0):
                raise ValueError(
                    "wave_activity must contain "
                    "only positive CWEF rows: "
                    f"point_id={summary.point_id}"
                )

        zero_count = (
            n_days - active_values.size
        )

        if zero_count:
            all_values = np.concatenate(
                (
                    active_values,
                    np.zeros(
                        zero_count,
                        dtype=np.float64,
                    ),
                )
            )
        else:
            all_values = active_values.copy()

        mean_cwef = float(
            np.mean(all_values)
        )

        if self._storm_threshold is None:
            threshold = float(
                np.quantile(
                    all_values,
                    self._storm_percentile
                    / 100.0,
                )
            )
        else:
            threshold = (
                self._storm_threshold
            )

        storm_mask = (
            all_values > threshold
        )

        e_storm_mjm = float(
            np.sum(
                all_values[storm_mask]
            )
            * self._dt_seconds
            / 1e6
        )

        n_storm_days = int(
            np.count_nonzero(storm_mask)
        )

        k_dir, top3 = (
            self._directional_concentration(
                activity
            )
        )

        if mean_cwef == 0.0:
            cv = 0.0
        elif all_values.size < 2:
            cv = None
        else:
            cv = float(
                np.std(
                    all_values,
                    ddof=1,
                )
                / mean_cwef
            )

        return {
            "point_id": int(
                summary.point_id
            ),
            "mean_CWEF_Wm": round(
                mean_cwef,
                3,
            ),
            "E_storm_MJm": round(
                e_storm_mjm,
                3,
            ),
            "storm_threshold_Wm": round(
                threshold,
                3,
            ),
            "storm_percentile": (
                self._storm_percentile
            ),
            "K_dir": round(
                k_dir,
                4,
            ),
            "CV": (
                None
                if cv is None
                else round(cv, 4)
            ),
            "n_days": n_days,
            "n_storm_days": (
                n_storm_days
            ),
            "top3_sectors": (
                None
                if not top3
                else ",".join(top3)
            ),
        }

    @staticmethod
    def _directional_concentration(
        activity: list[WaveActivityRow],
    ) -> tuple[float, list[str]]:
        if not activity:
            return 0.0, []

        sector_sums: dict[str, float] = {
            label: 0.0
            for label in SECTOR_LABELS
        }

        for row in activity:
            sector = _direction_to_sector(
                row.wind_azimuth_deg
            )
            sector_sums[sector] += float(
                row.cwef_wm
            )

        total = float(
            sum(sector_sums.values())
        )

        if total <= 0.0:
            return 0.0, []

        top3 = _top3_adjacent_sectors(
            sector_sums
        )

        concentration = float(
            sum(
                sector_sums[label]
                for label in top3
            )
            / total
        )

        return concentration, top3
