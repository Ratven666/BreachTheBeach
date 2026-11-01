import heapq
import math

from loguru import logger

from shapely.geometry import Point
from shapely.ops import unary_union

from .geometry import (
    Geometry,
    InvalidScenarioError,
    geometry_parts,
    is_connected,
    neighbours,
)


# ============================================================
# ЯКОРЬ И ИНТЕНСИВНОСТЬ ИЗМЕНЕНИЙ
# ============================================================

ANCHOR_CELL = (0, 0)

# Верхний предел удаления относительно числа клеток родителя.
LOCAL_MUTATION_FRACTION = 0.25

# Дополнительный абсолютный предел.
LOCAL_MUTATION_MAX_CELLS = 96

# Число попыток получить допустимую новую форму.
LOCAL_MUTATION_ATTEMPTS = 40

# Запрашиваемое удаление:
# от 50% рассчитанного лимита до полного лимита.
MIN_REMOVAL_FRACTION = 0.5

# Попытка принимается только если удалось удалить
# не менее 70% запрошенного числа клеток.
MIN_REALIZED_REMOVAL_FRACTION = 0.7


class AnchoredGeometry(Geometry):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.anchor_cell = ANCHOR_CELL
        self.anchor_point = Point(0, 0)

        if ANCHOR_CELL not in self.cells:
            raise ValueError(
                "Ячейка исходной точки недопустима. "
                "Она может полностью лежать на суше, "
                "касаться берега в режиме island "
                "или не помещаться в радиус. "
                "Якорь автоматически не переносится."
            )

        if not self.cells[ANCHOR_CELL].contains(
            self.anchor_point
        ):
            raise ValueError(
                "Исходная точка должна находиться "
                "внутри центральной ячейки"
            )

    def valid(self, cells):
        return (
            ANCHOR_CELL in cells
            and super().valid(cells)
        )

    def scenario(self, cells):
        result = super().scenario(cells)

        if cells and not result.fill.contains(
            self.anchor_point
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

        # Последовательность действий последней успешной мутации:
        # ("remove", cell), ("add", cell).
        self.last_trace = ()

        for name, value in {
            "LOCAL_MUTATION_FRACTION": (
                LOCAL_MUTATION_FRACTION
            ),
            "MIN_REMOVAL_FRACTION": (
                MIN_REMOVAL_FRACTION
            ),
            "MIN_REALIZED_REMOVAL_FRACTION": (
                MIN_REALIZED_REMOVAL_FRACTION
            ),
        }.items():
            if not 0 < value <= 1:
                raise ValueError(
                    f"{name} должно лежать в (0, 1]"
                )

        for name, value in {
            "LOCAL_MUTATION_MAX_CELLS": (
                LOCAL_MUTATION_MAX_CELLS
            ),
            "LOCAL_MUTATION_ATTEMPTS": (
                LOCAL_MUTATION_ATTEMPTS
            ),
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{name} должно быть целым > 0"
                )

    def change_limit(self, parent):
        return max(
            1,
            min(
                LOCAL_MUTATION_MAX_CELLS,
                math.ceil(
                    len(parent)
                    * LOCAL_MUTATION_FRACTION
                ),
            ),
        )

    # ========================================================
    # ПЕРВОЕ РЕШЕНИЕ
    # ========================================================

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

        destination = None

        while queue:
            cost, cell = heapq.heappop(queue)

            if cost != distance[cell]:
                continue

            if geometry.units[cell] == 1:
                destination = cell
                break

            for other in neighbours(cell):
                if other not in geometry.cells:
                    continue

                new_cost = (
                    cost
                    + geometry.units[other]
                )

                if (
                    new_cost <= self.cfg.target_units
                    and new_cost
                    < distance.get(other, math.inf)
                ):
                    distance[other] = new_cost
                    previous[other] = cell

                    heapq.heappush(
                        queue,
                        (new_cost, other),
                    )

        if destination is None:
            raise ValueError(
                "Невозможно связать якорь с берегом "
                "в пределах площади и радиуса"
            )

        path = set()

        while destination is not None:
            path.add(destination)
            destination = previous[destination]

        return frozenset(path)

    def _simple_add(self, cell, cells):
        """Консервативная локальная защита от замыкания воды."""
        i, j = cell
        side = self.cfg.cell_side_m

        ring = {
            (i + dx, j + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if dx or dy
        }

        free = set()

        for other in ring:
            if other in cells:
                continue

            if other not in self.center_water:
                self.center_water[other] = (
                    not self.geometry.local_land.covers(
                        Point(
                            other[0] * side,
                            other[1] * side,
                        )
                    )
                )

            if self.center_water[other]:
                free.add(other)

        if not free:
            return True

        visited = set()
        pending = [min(free)]

        while pending:
            current = pending.pop()

            if current in visited:
                continue

            visited.add(current)

            pending.extend(
                other
                for other in neighbours(current)
                if other in free
                and other not in visited
            )

        return len(visited) == len(free)

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

    def _initial_grow(self, core):
        geometry = self.geometry
        cells = set(core)

        units = sum(
            geometry.units[cell]
            for cell in cells
        )

        frontier = {
            other
            for cell in cells
            for other in neighbours(cell)
            if other in geometry.cells
            and other not in cells
        }

        while units < self.cfg.target_units:
            options = [
                cell
                for cell in sorted(frontier)
                if geometry.units[cell]
                <= self.cfg.target_units - units
            ]

            def priority(cell):
                radial = (
                    cell[0]**2 + cell[1]**2
                ) / max(
                    1,
                    self.cfg.target_units / 2,
                )

                shared = sum(
                    other in cells
                    for other in neighbours(cell)
                )

                return (
                    radial
                    - 0.3 * shared
                    + self.rng.random() * 0.7
                )

            options.sort(key=priority)

            selected = next(
                (
                    cell
                    for cell in options
                    if self._simple_add(cell, cells)
                ),
                None,
            )

            if selected is None:
                return None

            cells.add(selected)
            units += geometry.units[selected]

            frontier.discard(selected)

            frontier.update(
                other
                for other in neighbours(selected)
                if other in geometry.cells
                and other not in cells
            )

        if units != self.cfg.target_units:
            return None

        return frozenset(cells)

    def initial(self):
        if self.reference is not None:
            return self.reference

        core = self._seed_core()

        for attempt in range(
            self.cfg.construction_attempts
        ):
            candidate = self._initial_grow(core)

            if self._checked(candidate):
                self.reference = candidate

                logger.info(
                    "Первый якорный намыв построен | "
                    "квадратов={} | попыток={}",
                    len(candidate),
                    attempt + 1,
                )

                return candidate

        raise RuntimeError(
            "Не удалось построить первый якорный намыв"
        )

    # ========================================================
    # ТЕКУЩИЙ МОРСКОЙ ФРОНТ
    # ========================================================

    def _sea_frontiers(self, cells):
        """Возвращает:
        - клетки намыва, которые можно рассматривать для удаления;
        - внешние клетки, которые можно рассматривать для добавления.

        Обе группы должны иметь общую грань с морским фронтом.
        """
        geometry = self.geometry

        fill = geometry._fill(cells)

        land = geometry._q(
            geometry.local_land.union(fill)
        )

        water = geometry._q(
            geometry.frame.difference(land)
        )

        # Открытая вода должна иметь выход через участок
        # внешней границы локальной области.
        # Касание одной точкой выходом не считается.
        sea = geometry._q(
            unary_union(
                [
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
                ]
            )
        )

        if sea.is_empty:
            return [], []

        sea_front = geometry._q(
            fill.boundary.intersection(
                sea.boundary
            )
        )

        length_tolerance = max(
            geometry.precision * 0.5,
            1e-9,
        )

        boundary_cells = [
            cell
            for cell in sorted(cells)
            if any(
                other not in cells
                for other in neighbours(cell)
            )
        ]

        removable = [
            cell
            for cell in boundary_cells
            if cell != ANCHOR_CELL
            and (
                geometry.cells[cell]
                .boundary
                .intersection(sea_front)
                .length
                > length_tolerance
            )
        ]

        outside = {
            other
            for cell in boundary_cells
            for other in neighbours(cell)
            if other in geometry.cells
            and other not in cells
        }

        addable = [
            cell
            for cell in sorted(outside)
            if (
                geometry.cells[cell]
                .boundary
                .intersection(sea_front)
                .length
                > length_tolerance
            )
            and (
                geometry.cells[cell]
                .intersection(sea)
                .area
                > geometry.precision**2
            )
        ]

        return removable, addable

    # ========================================================
    # НАПРАВЛЕННОЕ СНЯТИЕ УЧАСТКА
    # ========================================================

    def _peel_with_trace(
        self,
        parent,
        number,
        angle=0.0,
    ):
        cells = set(parent)
        trace = []

        vx = math.cos(angle)
        vy = math.sin(angle)

        scale = max(
            1,
            math.sqrt(len(parent)),
        )

        for _ in range(number):
            options, _ = self._sea_frontiers(
                cells
            )

            removed = (
                set(parent) - cells
            )

            def priority(cell):
                adjacent_removed = sum(
                    other in removed
                    for other in neighbours(cell)
                )

                directional = (
                    cell[0] * vx
                    + cell[1] * vy
                ) / scale

                return (
                    directional
                    + 0.8 * adjacent_removed
                    + self.rng.random() * 0.15
                )

            options.sort(
                key=priority,
                reverse=True,
            )

            for cell in options:
                reduced = cells - {cell}

                if not is_connected(reduced):
                    continue

                if self.cfg.connection == "attached":
                    if not any(
                        self.geometry.units[other] == 1
                        for other in reduced
                    ):
                        continue

                cells = reduced
                trace.append(
                    ("remove", cell)
                )
                break

            else:
                break

        return frozenset(cells), trace

    def _peel(self, parent, number):
        # Совместимость с существующими тестами.
        cells, _ = self._peel_with_trace(
            parent,
            number,
        )
        return cells

    # ========================================================
    # НАПРАВЛЕННОЕ НАРАЩИВАНИЕ С ДРУГОЙ СТОРОНЫ
    # ========================================================

    def _regrow(
        self,
        core,
        forbidden,
        angle,
        donor,
    ):
        geometry = self.geometry
        cells = set(core)

        units = sum(
            geometry.units[cell]
            for cell in cells
        )

        trace = []

        vx = math.cos(angle)
        vy = math.sin(angle)

        scale = max(
            1,
            math.sqrt(
                self.cfg.target_units / 2
            ),
        )

        while units < self.cfg.target_units:
            _, options = self._sea_frontiers(
                cells
            )

            options = [
                cell
                for cell in options
                if cell not in forbidden
                and (
                    geometry.units[cell]
                    <= self.cfg.target_units - units
                )
            ]

            def priority(cell):
                projection = (
                    cell[0] * vx
                    + cell[1] * vy
                ) / scale

                radial = (
                    cell[0]**2 + cell[1]**2
                ) / max(
                    1,
                    self.cfg.target_units / 2,
                )

                shared = sum(
                    other in cells
                    for other in neighbours(cell)
                )

                donor_bonus = (
                    0.4 if cell in donor else 0.0
                )

                return (
                    -3.0 * projection
                    + 0.15 * radial
                    - 0.2 * shared
                    - donor_bonus
                    + self.rng.random() * 0.3
                )

            options.sort(key=priority)

            selected = next(
                (
                    cell
                    for cell in options
                    if self._simple_add(
                        cell,
                        cells,
                    )
                ),
                None,
            )

            if selected is None:
                return None, []

            cells.add(selected)
            units += geometry.units[selected]

            trace.append(
                ("add", selected)
            )

        return frozenset(cells), trace

    # ========================================================
    # МУТАЦИЯ
    # ========================================================

    def mutate(
        self,
        parent,
        donor=frozenset(),
    ):
        parent = frozenset(parent)
        self.last_trace = ()

        if not self.geometry.valid(parent):
            raise ValueError(
                "Недопустимый родитель"
            )

        limit = self.change_limit(parent)

        minimum_request = max(
            1,
            math.ceil(
                limit * MIN_REMOVAL_FRACTION
            ),
        )

        for attempt in range(
            LOCAL_MUTATION_ATTEMPTS
        ):
            requested = self.rng.randint(
                minimum_request,
                limit,
            )

            cut_angle = (
                self.rng.random()
                * 2
                * math.pi
            )

            core, cut_trace = (
                self._peel_with_trace(
                    parent,
                    requested,
                    cut_angle,
                )
            )

            removed = parent - core

            minimum_realized = max(
                1,
                math.ceil(
                    requested
                    * MIN_REALIZED_REMOVAL_FRACTION
                ),
            )

            if len(removed) < minimum_realized:
                continue

            # Рост преимущественно в другом секторе.
            growth_angle = (
                cut_angle
                + math.pi
                + self.rng.uniform(
                    -math.pi / 3,
                    math.pi / 3,
                )
            )

            candidate, add_trace = self._regrow(
                core=core,
                forbidden=removed,
                angle=growth_angle,
                donor=donor,
            )

            if candidate is None:
                continue

            final_removed = (
                parent - candidate
            )
            final_added = (
                candidate - parent
            )

            if (
                len(final_removed) > limit
                or len(final_added) > 2 * limit
            ):
                continue

            if not self._checked(candidate):
                continue

            self.last_trace = tuple(
                cut_trace + add_trace
            )

            logger.info(
                "МОРСКАЯ МУТАЦИЯ | "
                "удалено={} | добавлено={} | "
                "изменено относительно родителя={:.1%} | "
                "попытка={}",
                len(final_removed),
                len(final_added),
                (
                    len(final_removed)
                    + len(final_added)
                ) / len(parent),
                attempt + 1,
            )

            return candidate

        logger.warning(
            "Допустимая морская мутация не найдена. "
            "Родитель сохранён без перестроения. "
            "Лимит удаления={}",
            limit,
        )

        return parent

    def child(self, first, second):
        # Каждый неэлитный вариант пытается измениться.
        # Старое случайное пропускание мутации удалено.
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
        # Совместимость с существующими тестами.
        if self.reference is None:
            return self.initial()

        self.reference = self.mutate(
            self.reference,
            preferred,
        )

        return self.reference