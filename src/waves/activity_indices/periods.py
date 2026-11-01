from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .constants import DAYS_PER_YEAR, EPOCH, MAX_DAY_OFFSET


@dataclass(frozen=True, slots=True)
class Period:
    period_id: int
    start_day: int
    end_day: int
    is_complete: bool

    @property
    def n_days(self) -> int:
        return self.end_day - self.start_day + 1

    @property
    def mid_year(self) -> float:
        mid_day = 0.5 * (self.start_day + self.end_day + 1)
        return EPOCH.year + mid_day / DAYS_PER_YEAR


def day_to_date(day: int) -> date:
    return EPOCH + timedelta(days=int(day))


def date_to_day(value: date) -> int:
    return (value - EPOCH).days


def add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def build_periods(start_day: int, end_day: int, interval_years: int) -> list[Period]:
    """Строит периоды длиной interval_years календарных лет в целых днях.

    Начало отсчёта — start_day. Последний период обрезается по end_day
    и помечается is_complete=False, если он короче полного интервала.
    """
    if int(interval_years) < 1:
        raise ValueError("interval_years must be >= 1")
    if not 0 <= start_day <= end_day <= MAX_DAY_OFFSET:
        raise ValueError(
            f"Invalid day range: start_day={start_day}, end_day={end_day}, "
            f"allowed 0..{MAX_DAY_OFFSET}"
        )

    anchor = day_to_date(start_day)
    periods: list[Period] = []
    k = 0

    while True:
        p_start = date_to_day(add_years(anchor, k * interval_years))
        if p_start > end_day:
            break
        p_next = date_to_day(add_years(anchor, (k + 1) * interval_years))
        p_end = p_next - 1
        complete = p_end <= end_day
        periods.append(
            Period(
                period_id=k + 1,
                start_day=p_start,
                end_day=min(p_end, end_day),
                is_complete=complete,
            )
        )
        k += 1

    return periods
