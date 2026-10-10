from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from .calc import (
    aggregate_point,
    alongshore_load_wm,
    diffs_to_previous,
    r,
    series_stats,
    sign_of,
)
from .constants import (
    ALONGSHORE_SIGN_CONVENTION,
    EPOCH,
    MAX_ANGLE_DEG,
    TREND_ONLY_COMPLETE_PERIODS,
)
from .periods import build_periods
from .repository import IndexRepository, SourceActivityReader


@dataclass(frozen=True, slots=True)
class BuildStats:
    n_points: int
    n_periods: int
    start_day: int
    end_day: int
    output_path: Path


class ActivityIndexBuilder:
    """Строит отдельную БД индексов из wave_activity.db.

    start_day / end_day — целые дни от EPOCH (1940-01-01 = 0), включительно.
    end_day=None -> start_day + max(n_days) - 1 по wave_activity_summary.
    """

    def __init__(
        self,
        source_db: Path,
        output_db: Path,
        *,
        interval_years: int,
        start_day: int,
        end_day: int | None,
        max_angle_deg: float = MAX_ANGLE_DEG,
        trend_only_complete_periods: bool = TREND_ONLY_COMPLETE_PERIODS,
    ) -> None:
        if not 0.0 < max_angle_deg < 90.0:
            raise ValueError("max_angle_deg must be in (0, 90)")
        self.source_db = Path(source_db)
        self.output_db = Path(output_db)
        self.interval_years = int(interval_years)
        self.start_day = int(start_day)
        self.end_day = None if end_day is None else int(end_day)
        self.max_angle_deg = float(max_angle_deg)
        self.trend_only_complete = bool(trend_only_complete_periods)

    def run(self) -> BuildStats:
        tmp = self.output_db.with_name(self.output_db.name + ".tmp")
        for p in (tmp, Path(f"{tmp}-journal")):
            p.unlink(missing_ok=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)

        with SourceActivityReader(self.source_db) as src, IndexRepository(tmp) as out:
            points = src.read_points()
            if not points:
                raise ValueError("wave_points is empty")
            n_days_map = src.read_n_days()
            bounds = src.day_bounds()

            end_day = self.end_day
            if end_day is None:
                if not n_days_map:
                    raise ValueError("wave_activity_summary is empty; set end_day explicitly")
                end_day = self.start_day + max(n_days_map.values()) - 1
            if bounds is not None and (bounds[0] < self.start_day or bounds[1] > end_day):
                raise ValueError(
                    f"Activity days {bounds} are outside [{self.start_day}, {end_day}]"
                )
            mismatch = [i for i, n in n_days_map.items() if n != end_day - self.start_day + 1]
            if mismatch:
                logger.warning(
                    "{} points have n_days != analysis length {} (e.g. {})",
                    len(mismatch), end_day - self.start_day + 1, mismatch[:5],
                )

            periods = build_periods(self.start_day, end_day, self.interval_years)
            logger.info("Periods: {} (days {}..{})", len(periods), self.start_day, end_day)

            out.initialize()
            out.add_meta(
                {
                    "epoch": EPOCH.isoformat(),
                    "source_db": self.source_db.resolve(),
                    "interval_years": self.interval_years,
                    "start_day": self.start_day,
                    "end_day": end_day,
                    "max_angle_deg": self.max_angle_deg,
                    "trend_only_complete_periods": int(self.trend_only_complete),
                    "alongshore_sign_convention": ALONGSHORE_SIGN_CONVENTION,
                    "alongshore_method": "CWEF * tan(clip(theta)); proxy, CWEF is post-refraction",
                }
            )
            out.add_many("points", points)
            out.add_many(
                "periods",
                [(p.period_id, p.start_day, p.end_day, p.n_days, int(p.is_complete), p.mid_year)
                 for p in periods],
            )

            years = np.array([p.mid_year for p in periods])
            trend_mask = np.array([p.is_complete or not self.trend_only_complete for p in periods])

            for k, (pid, _lon, _lat, normal_az) in enumerate(points, start=1):
                days, wind, cwef = src.iter_activity(pid)
                along = alongshore_load_wm(cwef, wind, normal_az, self.max_angle_deg)
                items = aggregate_point(days, cwef, along, periods)
                diffs = diffs_to_previous(items)

                out.add_many(
                    "point_period_indices",
                    [
                        (
                            pid, p.period_id, it.n_active_days,
                            r(it.normal_mean_wm), r(it.alongshore_net_mean_wm),
                            r(it.alongshore_abs_mean_wm), r(it.alongshore_pos_mean_wm),
                            r(it.alongshore_neg_mean_wm),
                            r(d["d_normal_wm"]), r(d["d_normal_pct"]),
                            r(d["d_alongshore_abs_wm"]), r(d["d_alongshore_abs_pct"]),
                            r(d["d_alongshore_net_wm"]),
                        )
                        for p, it, d in zip(periods, items, diffs)
                    ],
                )

                sel = [it for it, m in zip(items, trend_mask) if m]
                yrs = years[trend_mask]
                out.add_many("point_trends", [self._trend_row(pid, sel, yrs)])

                if k % 50 == 0 or k == len(points):
                    logger.info("Points processed: {}/{}", k, len(points))

            out.commit()

        os.replace(tmp, self.output_db)
        logger.success("Index DB written: {}", self.output_db)
        return BuildStats(len(points), len(periods), self.start_day, end_day, self.output_db)

    @staticmethod
    def _trend_row(pid: int, items: list, years: np.ndarray) -> tuple:
        if not items:
            return (pid, 0) + (None,) * 12 + (None,)
        n = np.array([i.normal_mean_wm for i in items])
        a = np.array([i.alongshore_abs_mean_wm for i in items])
        net = np.array([i.alongshore_net_mean_wm for i in items])
        sn, sa, snet = series_stats(years, n), series_stats(years, a), series_stats(years, net)
        return (
            pid, len(items),
            r(sn["mean"]), r(sn["cv"]), r(sn["slope"]), r(sn["r2"]), r(sn["first_last_pct"]),
            r(sa["mean"]), r(sa["cv"]), r(sa["slope"]), r(sa["r2"]), r(sa["first_last_pct"]),
            r(snet["mean"]), r(snet["slope"]),
            sign_of(snet["mean"]),
        )
