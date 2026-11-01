"""Якорный намыв и направленный поиск с ограниченным возвратом из тупиков."""

import heapq
import math

from collections import Counter
from concurrent.futures import CancelledError
from dataclasses import dataclass
from fractions import Fraction

from shapely.geometry import Point
from shapely.ops import unary_union

from .geometry import (
    Geometry,
    InvalidScenarioError,
    geometry_parts,
    is_connected,
    neighbours,
)


ANCHOR_CELL = (0, 0)

# Конечная симметрическая разность относительно родителя.
# 10% изменённых позиций = 5% удалённых + 5% добавленных клеток.
MIN_CHANGED_CELL_FRACTION = 0.10
MAX_CHANGED_CELL_FRACTION = 0.25

# Допуски смещения в сторонах клетки.
# Для больших намывов масштабируются с sqrt(N).
BASE_BOUNDARY_SHIFT_CELLS = 3.0
BASE_CENTROID_SHIFT_CELLS = 1.0

BOUNDARY_SHIFT_SCALE = 0.20
CENTROID_SHIFT_SCALE = 0.08

# Полуширина фиксированной полосы изменения.
STRIP_HALF_WIDTH_FRACTION = 0.25
STRIP_MIN_HALF_WIDTH_CELLS = 2.0

# Ограничение изрезанности квадратного контура.
MAX_PERIMETER_GROWTH_FRACTION = 0.15

# Число рассматриваемых удалений и добавлений.
REMOVAL_CHOICES = 12
ADDITION_CHOICES = 12

# Максимум дорогих проверок пар на одном состоянии.
PAIR_CHECK_LIMIT = 32

# Сколько уровней альтернатив сохраняется для возврата.
BACKTRACK_DEPTH = 4

# Бюджет исследования одной задачи мутации.
MAX_SEARCH_NODES = 220

MAX_STAGNANT_MOVES = 8
MAX_PATH_MOVES_FRACTION = 0.50

# Совместимость с внешними импортами.
LOCAL_MUTATION_FRACTION = MAX_CHANGED_CELL_FRACTION
LOCAL_MUTATION_MAX_CELLS = None
LOCAL_MUTATION_ATTEMPTS = PAIR_CHECK_LIMIT


@dataclass(frozen=True)
class MutationPlan:
    azimuth_deg: float
    strip_position: float
    strength: float

    def validate(self):
        values = (
            self.azimuth_deg,
            self.strip_position,
            self.strength,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Параметры плана должны быть конечными"
            )

        if not (
            0 <= self.strip_position <= 1
            and 0 <= self.strength <= 1
        ):
            raise ValueError(
                "strip_position и strength "
                "должны лежать в [0, 1]"
            )


def floor_part(count, fraction):
    value = Fraction(str(fraction))

    return (
        count
        * value.numerator
        // value.denominator
    )


def ceil_part(count, fraction):
    value = Fraction(str(fraction))

    return (
        count * value.numerator
        + value.denominator - 1
    ) // value.denominator


def perimeter(cells):
    """Периметр объединения квадратов в сторонах клетки."""
    return sum(
        other not in cells
        for cell in cells
        for other in neighbours(cell)
    )


class AnchoredGeometry(Geometry):
    def __init__(self, cfg):
        self.fixed_cell_count = None

        super().__init__(cfg)

        self.anchor_cell = ANCHOR_CELL
        self.anchor_point = Point(0, 0)

        if (
            ANCHOR_CELL not in self.cells
            or not self.cells[ANCHOR_CELL].contains(
                self.anchor_point
            )
        ):
            raise ValueError(
                "Якорь (0, 0) не находится "
                "внутри допустимой клетки"
            )

    def lock_cell_count(self, count):
        if type(count) is not int or count < 2:
            raise ValueError(
                "Нужно минимум два квадрата"
            )

        if self.fixed_cell_count not in (None, count):
            raise ValueError(
                "Количество клеток уже зафиксировано"
            )

        self.fixed_cell_count = count

    def valid_with_fill(self, cells, fill):
        if (
            len(cells) < 2
            or ANCHOR_CELL not in cells
            or not set(cells).issubset(self.cells)
        ):
            return False

        if self.fixed_cell_count is None:
            units = sum(
                self.units[cell]
                for cell in cells
            )

            if units != self.cfg.target_units:
                return False

        elif len(cells) != self.fixed_cell_count:
            return False

        if (
            not is_connected(cells)
            or not fill.contains(self.anchor_point)
        ):
            return False

        if self.cfg.connection == "attached":
            if not any(
                self.units[cell] == 1
                for cell in cells
            ):
                return False

        if any(
            polygon.interiors
            for polygon in geometry_parts(
                fill,
                {"Polygon"},
            )
        ):
            return False

        land = self._q(
            self.local_land.union(fill)
        )

        lakes = self._q(
            self._closed_water(land).difference(
                self.baseline_lakes
            )
        )

        return lakes.area <= self.precision**2

    def valid(self, cells):
        cells = frozenset(cells)

        return (
            bool(cells)
            and cells.issubset(self.cells)
            and self.valid_with_fill(
                cells,
                self._fill(cells),
            )
        )

    def scenario(self, cells):
        result = super().scenario(cells)

        if (
            cells
            and not result.fill.contains(
                self.anchor_point
            )
        ):
            raise InvalidScenarioError(
                "Якорь не находится внутри намыва"
            )

        return result


class GeneticBuilder:
    def __init__(self, cfg, geometry, rng):
        self.cfg = cfg
        self.geometry = geometry
        self.rng = rng

        self.reference = None
        self.center_water = {}

        self.last_trace = ()
        self.last_stats = {}

        self.plan = None
        self.cancel_event = None

        if not (
            0
            < MIN_CHANGED_CELL_FRACTION
            <= MAX_CHANGED_CELL_FRACTION
            <= 1
        ):
            raise ValueError(
                "Неверные доли изменения"
            )

        for value in (
            REMOVAL_CHOICES,
            ADDITION_CHOICES,
            PAIR_CHECK_LIMIT,
            BACKTRACK_DEPTH,
            MAX_SEARCH_NODES,
            MAX_STAGNANT_MOVES,
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Лимиты поиска должны быть "
                    "целыми и > 0"
                )

    def _check_cancelled(self):
        if (
            self.cancel_event is not None
            and self.cancel_event.is_set()
        ):
            raise CancelledError(
                "Поиск мутации отменён"
            )

    def change_limit(self, parent):
        count = floor_part(
            len(parent),
            MAX_CHANGED_CELL_FRACTION,
        )

        return count - count % 2

    def minimum_change(self, parent):
        count = max(
            2,
            ceil_part(
                len(parent),
                MIN_CHANGED_CELL_FRACTION,
            ),
        )

        return count + count % 2

    def boundary_shift(self, parent):
        return max(
            BASE_BOUNDARY_SHIFT_CELLS,
            BOUNDARY_SHIFT_SCALE
            * math.sqrt(len(parent)),
        )

    def centroid_shift(self, parent):
        return max(
            BASE_CENTROID_SHIFT_CELLS,
            CENTROID_SHIFT_SCALE
            * math.sqrt(len(parent)),
        )

    def _seed_core(self):
        geometry = self.geometry

        if (
            self.cfg.connection == "island"
            or geometry.units[ANCHOR_CELL] == 1
        ):
            return frozenset({ANCHOR_CELL})

        distance = {
            ANCHOR_CELL: geometry.units[ANCHOR_CELL]
        }

        previous = {
            ANCHOR_CELL: None
        }

        queue = [
            (
                distance[ANCHOR_CELL],
                ANCHOR_CELL,
            )
        ]

        while queue:
            cost, cell = heapq.heappop(queue)

            if cost != distance[cell]:
                continue

            if geometry.units[cell] == 1:
                path = set()

                while cell is not None:
                    path.add(cell)
                    cell = previous[cell]

                return frozenset(path)

            for other in neighbours(cell):
                if other not in geometry.cells:
                    continue

                value = (
                    cost + geometry.units[other]
                )

                if (
                    value <= self.cfg.target_units
                    and value
                    < distance.get(other, math.inf)
                ):
                    distance[other] = value
                    previous[other] = cell

                    heapq.heappush(
                        queue,
                        (value, other),
                    )

        raise ValueError(
            "Нельзя соединить якорь с берегом "
            "в пределах начальной площади"
        )

    def _simple_add(self, cell, cells):
        free = set()

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (dx or dy):
                    continue

                other = (
                    cell[0] + dx,
                    cell[1] + dy,
                )

                if other in cells:
                    continue

                if other not in self.center_water:
                    self.center_water[other] = (
                        not self.geometry.local_land.covers(
                            Point(
                                other[0]
                                * self.cfg.cell_side_m,
                                other[1]
                                * self.cfg.cell_side_m,
                            )
                        )
                    )

                if self.center_water[other]:
                    free.add(other)

        return (
            not free
            or is_connected(free)
        )

    def _initial_grow(self, core):
        geometry = self.geometry
        cells = set(core)

        units = sum(
            geometry.units[cell]
            for cell in cells
        )

        front = {
            other
            for cell in cells
            for other in neighbours(cell)
            if other in geometry.cells
            and other not in cells
        }

        while units < self.cfg.target_units:
            self._check_cancelled()

            options = sorted(
                cell
                for cell in front
                if geometry.units[cell]
                <= self.cfg.target_units - units
            )

            scores = {
                cell: (
                    (
                        cell[0]**2 + cell[1]**2
                    )
                    / max(
                        1,
                        self.cfg.target_units / 2,
                    )
                    - 0.3 * sum(
                        other in cells
                        for other in neighbours(cell)
                    )
                    + self.rng.random() * 0.7
                )
                for cell in options
            }

            options.sort(
                key=lambda cell: (
                    scores[cell],
                    cell,
                )
            )

            cell = next(
                (
                    candidate
                    for candidate in options
                    if self._simple_add(
                        candidate,
                        cells,
                    )
                ),
                None,
            )

            if cell is None:
                return None

            cells.add(cell)
            units += geometry.units[cell]

            front.discard(cell)

            front.update(
                other
                for other in neighbours(cell)
                if other in geometry.cells
                and other not in cells
            )

        return (
            frozenset(cells)
            if units == self.cfg.target_units
            else None
        )

    def _checked(self, cells):
        if (
            cells is None
            or not self.geometry.valid(cells)
        ):
            return False

        try:
            self.geometry.scenario(cells)
            return True

        except InvalidScenarioError:
            return False

    def initial(self):
        if self.reference is not None:
            return self.reference

        core = self._seed_core()

        for _ in range(
            self.cfg.construction_attempts
        ):
            cells = self._initial_grow(core)

            if self._checked(cells):
                self.geometry.lock_cell_count(
                    len(cells)
                )

                self.reference = cells

                return cells

        raise RuntimeError(
            "Не удалось построить "
            "первый якорный намыв"
        )

    def _sea_frontiers(self, cells, fill=None):
        geometry = self.geometry

        if fill is None:
            fill = geometry._fill(cells)

        water = geometry._q(
            geometry.frame.difference(
                geometry._q(
                    geometry.local_land.union(fill)
                )
            )
        )

        sea = geometry._q(
            unary_union([
                polygon
                for polygon in geometry_parts(
                    water,
                    {"Polygon"},
                )
                if (
                    polygon.boundary.intersection(
                        geometry.frame.boundary
                    ).length
                    > geometry.precision
                )
            ])
        )

        if sea.is_empty:
            return [], []

        front = geometry._q(
            fill.boundary.intersection(
                sea.boundary
            )
        )

        boundary = sorted(
            cell
            for cell in cells
            if any(
                other not in cells
                for other in neighbours(cell)
            )
        )

        outside = sorted({
            other
            for cell in boundary
            for other in neighbours(cell)
            if other in geometry.cells
            and other not in cells
        })

        epsilon = max(
            geometry.precision * 0.5,
            1e-9,
        )

        removable = [
            cell
            for cell in boundary
            if cell != ANCHOR_CELL
            and (
                geometry.cells[cell]
                .boundary.intersection(front)
                .length
                > epsilon
            )
        ]

        addable = [
            cell
            for cell in outside
            if (
                geometry.cells[cell]
                .boundary.intersection(front)
                .length
                > epsilon
            )
            and (
                geometry.cells[cell]
                .intersection(sea)
                .area
                > geometry.precision**2
            )
        ]

        return removable, addable

    def _alternatives(self, state, context):
        (
            cells,
            fill,
            trace,
            boundary_length,
            stagnant,
        ) = state

        (
            parent,
            donor,
            vector,
            strip_center,
            strip_width,
            band,
            parent_fill,
            rejected,
        ) = context

        dx, dy = vector

        def projection(cell):
            return cell[0] * dx + cell[1] * dy

        def in_strip(cell):
            transverse = (
                -cell[0] * dy + cell[1] * dx
            )

            return (
                abs(transverse - strip_center)
                <= strip_width + 1e-12
            )

        removable, addable = self._sea_frontiers(
            cells,
            fill,
        )

        removable = [
            cell
            for cell in removable
            if in_strip(cell)
        ]

        addable = [
            cell
            for cell in addable
            if in_strip(cell)
        ]

        # Шум влияет на числовую оценку,
        # а не только разрывает равенство ключей.
        noise = {
            cell: self.rng.uniform(-1, 1)
            for cell in sorted(
                set(removable) | set(addable)
            )
        }

        removable.sort(
            key=lambda cell: (
                cell not in parent,
                projection(cell) + noise[cell],
                cell,
            )
        )

        pairs = []

        before = len(parent ^ cells)

        limit = perimeter(parent) * (
            1 + MAX_PERIMETER_GROWTH_FRACTION
        )

        for removed in removable[
            :REMOVAL_CHOICES
        ]:
            self._check_cancelled()

            core = cells - {removed}

            if not is_connected(core):
                rejected["disconnected_core"] += 1
                continue

            options = [
                cell
                for cell in addable
                if any(
                    other in core
                    for other in neighbours(cell)
                )
            ]

            options.sort(
                key=lambda cell: (
                    cell in parent,
                    -projection(cell) + noise[cell],
                    cell,
                )
            )

            shared_removed = sum(
                other in cells
                for other in neighbours(removed)
            )

            for added in options[
                :ADDITION_CHOICES
            ]:
                displacement = (
                    projection(added)
                    - projection(removed)
                )

                trial = frozenset(
                    core | {added}
                )

                changed = len(
                    parent ^ trial
                )

                shared_added = sum(
                    other in core
                    for other in neighbours(added)
                )

                new_length = (
                    boundary_length
                    + 2 * (
                        shared_removed
                        - shared_added
                    )
                )

                if (
                    displacement <= 1e-10
                    or changed < before
                    or changed > self.change_limit(parent)
                ):
                    rejected["direction_or_budget"] += 1
                    continue

                if new_length > limit:
                    rejected["perimeter"] += 1
                    continue

                # Взвешенная случайная перестановка:
                # хорошие пары вероятнее,
                # но не выбираются безальтернативно.
                compactness = (
                    new_length - boundary_length
                )

                weight = (
                    1
                    if changed > before
                    else 0.15
                )

                weight *= math.exp(
                    -0.35 * max(
                        -4,
                        compactness,
                    )
                )

                if added in donor:
                    weight *= 1.5

                weight *= (
                    1
                    + min(
                        displacement
                        / max(
                            1,
                            math.sqrt(len(parent)),
                        ),
                        2,
                    )
                )

                priority = (
                    -math.log(
                        max(
                            self.rng.random(),
                            1e-15,
                        )
                    )
                    / weight
                )

                pairs.append((
                    priority,
                    removed,
                    added,
                    trial,
                    new_length,
                    changed,
                ))

        pairs.sort(
            key=lambda row: (
                row[0],
                row[1],
                row[2],
            )
        )

        geometry = self.geometry
        side = self.cfg.cell_side_m

        for (
            _,
            removed,
            added,
            trial,
            new_length,
            changed,
        ) in pairs[:PAIR_CHECK_LIMIT]:
            self._check_cancelled()

            new_fill = geometry._q(
                fill.difference(
                    geometry.cells[removed]
                ).union(
                    geometry.cells[added]
                )
            )

            if (
                parent_fill.centroid.distance(
                    new_fill.centroid
                )
                > self.centroid_shift(parent) * side

                or new_fill.boundary.difference(
                    band
                ).length
                > geometry.precision

                or parent_fill.boundary.difference(
                    new_fill.boundary.buffer(
                        self.boundary_shift(parent)
                        * side
                    )
                ).length
                > geometry.precision
            ):
                rejected["displacement"] += 1
                continue

            if not geometry.valid_with_fill(
                trial,
                new_fill,
            ):
                rejected["topology"] += 1
                continue

            new_trace = trace + (
                ("remove", removed),
                ("add", added),
            )

            new_stagnant = (
                stagnant + 1
                if changed == before
                else 0
            )

            yield (
                trial,
                new_fill,
                new_trace,
                new_length,
                new_stagnant,
            )

    def mutate(self, parent, donor=frozenset()):
        parent = frozenset(parent)

        self.last_trace = ()

        self.last_stats = {
            "status": "searching",
            "changed_cell_count": 0,
        }

        self._check_cancelled()

        if not self.geometry.valid(parent):
            raise ValueError(
                "Недопустимый родитель"
            )

        self.geometry.lock_cell_count(
            len(parent)
        )

        minimum = self.minimum_change(parent)
        maximum = self.change_limit(parent)

        if maximum < minimum:
            self.last_stats.update(
                status="discrete_limit",
                minimum=minimum,
                maximum=maximum,
            )

            return parent

        plan = self.plan or MutationPlan(
            self.rng.uniform(0, 360),
            self.rng.random(),
            self.rng.random(),
        )

        plan.validate()

        angle = math.radians(
            plan.azimuth_deg
        )

        vector = (
            math.sin(angle),
            math.cos(angle),
        )

        dx, dy = vector

        cross = [
            -cell[0] * dy + cell[1] * dx
            for cell in sorted(parent)
        ]

        low = min(cross)
        high = max(cross)

        width = max(
            STRIP_MIN_HALF_WIDTH_CELLS,
            (high - low)
            * STRIP_HALF_WIDTH_FRACTION,
        )

        center = (
            low
            + (high - low)
            * (
                0.2
                + 0.6 * plan.strip_position
            )
        )

        target = (
            minimum
            + 2 * round(
                (maximum - minimum)
                / 2
                * plan.strength
            )
        )

        parent_fill = self.geometry._fill(
            parent
        )

        band = parent_fill.boundary.buffer(
            self.boundary_shift(parent)
            * self.cfg.cell_side_m
        )

        rejected = Counter()

        context = (
            parent,
            frozenset(donor) - parent,
            vector,
            center,
            width,
            band,
            parent_fill,
            rejected,
        )

        root = (
            parent,
            parent_fill,
            (),
            perimeter(parent),
            0,
        )

        stack = [
            iter(
                self._alternatives(
                    root,
                    context,
                )
            )
        ]

        seen = {parent}
        best = None

        nodes = 0
        backtracks = 0

        path_limit = max(
            1,
            ceil_part(
                len(parent),
                MAX_PATH_MOVES_FRACTION,
            ),
        )

        while (
            stack
            and nodes < MAX_SEARCH_NODES
        ):
            self._check_cancelled()

            state = next(
                stack[-1],
                None,
            )

            if state is None:
                stack.pop()
                backtracks += 1
                continue

            nodes += 1

            (
                cells,
                fill,
                trace,
                length,
                stagnant,
            ) = state

            if cells in seen:
                continue

            seen.add(cells)

            changed = len(
                parent ^ cells
            )

            if (
                changed >= minimum
                and (
                    best is None
                    or changed
                    > len(parent ^ best[0])
                )
            ):
                if self._checked(cells):
                    best = state

                    if changed >= target:
                        break

            if (
                len(trace) // 2 < path_limit
                and stagnant < MAX_STAGNANT_MOVES
            ):
                stack.append(
                    iter(
                        self._alternatives(
                            state,
                            context,
                        )
                    )
                )

                if len(stack) > BACKTRACK_DEPTH:
                    del stack[0]

        self.last_stats = {
            "status": (
                "changed"
                if best is not None
                else "no_feasible_mutation"
            ),
            "changed_cell_count": (
                len(parent ^ best[0])
                if best is not None
                else 0
            ),
            "minimum": minimum,
            "maximum": maximum,
            "target": target,

            "addition_azimuth_deg": (
                plan.azimuth_deg % 360
            ),
            "removal_azimuth_deg": (
                (plan.azimuth_deg + 180) % 360
            ),

            "direction_vector": vector,
            "strip_center_cells": center,
            "strip_half_width_cells": width,

            "search_nodes": nodes,
            "backtracks": backtracks,
            "rejections": dict(rejected),

            "cell_count": len(parent),
            "boundary_shift_cells": (
                self.boundary_shift(parent)
            ),
            "centroid_shift_cells": (
                self.centroid_shift(parent)
            ),
        }

        if best is None:
            return parent

        (
            cells,
            _,
            self.last_trace,
            length,
            _,
        ) = best

        self.last_stats.update(
            move_count=len(self.last_trace) // 2,

            perimeter_cells=length,
            parent_perimeter_cells=perimeter(parent),

            removed_cell_count=len(parent - cells),
            added_cell_count=len(cells - parent),

            credited_area_m2=(
                sum(
                    self.geometry.units[cell]
                    for cell in cells
                )
                * self.cfg.cell_side_m**2
                / 2
            ),
        )

        return cells

    def child(self, first, second):
        donor = (
            second
            if self.rng.random()
            < self.cfg.crossover_probability
            else frozenset()
        )

        return self.mutate(
            first,
            donor,
        )

    def build(
        self,
        preferred=frozenset(),
        core=frozenset(),
    ):
        if self.reference is None:
            return self.initial()

        self.reference = self.mutate(
            self.reference,
            preferred,
        )

        return self.reference