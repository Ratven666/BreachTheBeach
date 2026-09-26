from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

from .schema import COORD_SCALE, EPOCH, MAX_DAY_OFFSET, VALUE_SCALE


def _scaled_int(value: float, scale: int) -> int:
    return int((Decimal(str(value)) * scale).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def encode_coordinate(value: float) -> int:
    return _scaled_int(value, COORD_SCALE)


def decode_coordinate(value: int) -> float:
    return value / COORD_SCALE


def encode_value(value: float | None) -> int | None:
    return None if value is None else _scaled_int(value, VALUE_SCALE)


def decode_value(value: int | None) -> float | None:
    return None if value is None else value / VALUE_SCALE


def encode_date(value: date | str) -> int:
    parsed = date.fromisoformat(value) if isinstance(value, str) else value
    offset = (parsed - EPOCH).days
    if not 0 <= offset <= MAX_DAY_OFFSET:
        raise ValueError(f"Date {parsed.isoformat()} is outside compact database range")
    return offset


def decode_date(day: int) -> date:
    if not 0 <= day <= MAX_DAY_OFFSET:
        raise ValueError(f"Invalid compact day offset: {day}")
    return EPOCH + timedelta(days=day)
