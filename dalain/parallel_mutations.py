"""Постоянный spawn-пул.

Размер нового поколения может быть меньше запроса.
Копии родителя и точные дубликаты не возвращаются.
"""

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

from dataclasses import dataclass, replace
from collections import Counter

import hashlib
import multiprocessing as mp
import os
import pickle
import random
import signal
import threading
import time
import traceback

from loguru import logger

from .anchored import MutationPlan


MUTATION_BACKEND = "process"

_cpu_count = (
    getattr(
        os,
        "process_cpu_count",
        os.cpu_count,
    )()
    or 1
)

if hasattr(os, "sched_getaffinity"):
    _cpu_count = min(
        _cpu_count,
        len(os.sched_getaffinity(0)),
    )

MUTATION_WORKERS = max(
    1,
    min(
        8,
        _cpu_count,
    ),
)

MAX_PENDING_PER_WORKER = 2
PROGRESS_INTERVAL_S = 5.0

DIRECTION_SECTORS = 8
MAX_GENERATION_ROUNDS = 2
MIN_PAIR_DISTANCE = 0.02


@dataclass(frozen=True)
class MutationTask:
    index: int
    seed: int

    parent: frozenset
    donor: frozenset

    plan: MutationPlan


@dataclass
class MutationResult:
    index: int
    seed: int

    cells: frozenset
    seconds: float
    stats: dict

    error: str | None = None
    fatal: bool = False


class InsufficientDiversityError(RuntimeError):
    """Совместимость импортов.

    generate() больше не возбуждает это исключение.
    """


def cell_distance(first, second):
    if len(first) != len(second):
        raise ValueError(
            "Нужно одинаковое число клеток"
        )

    if not first:
        return 0.0

    return (
        len(first ^ second)
        / (2 * len(first))
    )


def select_diverse(results, count, threshold):
    if not results or count == 0:
        return []

    remaining = sorted(
        results,
        key=lambda result: result.index,
    )

    selected = [
        remaining.pop(0)
    ]

    distances = {
        result.index: cell_distance(
            result.cells,
            selected[0].cells,
        )
        for result in remaining
    }

    while remaining and len(selected) < count:
        row = max(
            remaining,
            key=lambda result: (
                distances[result.index],
                -result.index,
            ),
        )

        if distances[row.index] + 1e-12 < threshold:
            break

        selected.append(row)
        remaining.remove(row)

        for other in remaining:
            distances[other.index] = min(
                distances[other.index],
                cell_distance(
                    other.cells,
                    row.cells,
                ),
            )

    return selected


def make_task(
    index,
    round_index,
    parent,
    donor,
    seed,
    rotation,
):
    digest = hashlib.sha256(
        (
            f"dalain:{seed}:"
            f"{round_index}:{index}"
        ).encode()
    ).digest()

    task_seed = int.from_bytes(
        digest[:8],
        "big",
    )

    rng = random.Random(
        task_seed
    )

    sector = (
        index + rotation
    ) % DIRECTION_SECTORS

    azimuth = (
        sector
        + 0.5
        + rng.uniform(-0.3, 0.3)
    ) * 360 / DIRECTION_SECTORS

    position = (
        (
            index // DIRECTION_SECTORS
            + round_index
        )
        % 3
    ) / 2

    strength = (
        0.2,
        0.5,
        0.85,
    )[
        (index + round_index) % 3
    ]

    return MutationTask(
        index=index,
        seed=task_seed,
        parent=parent,
        donor=donor,
        plan=MutationPlan(
            azimuth % 360,
            position,
            strength,
        ),
    )


def _geometry_snapshot(geometry):
    state = dict(
        vars(geometry)
    )

    metadata = (
        geometry.crs.to_wkt(),
        geometry.geod.initstring,
    )

    for name in (
        "crs",
        "geod",
        "to_metric",
        "to_lonlat",
        "line_tree",
    ):
        state.pop(name, None)

    return pickle.dumps(
        (state, metadata),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _restore_geometry(
    geometry_type,
    snapshot,
):
    from pyproj import (
        CRS,
        Geod,
        Transformer,
    )

    from shapely import STRtree

    state, metadata = pickle.loads(snapshot)
    wkt, geod_init = metadata

    geometry = geometry_type.__new__(
        geometry_type
    )

    geometry.__dict__.update(
        state
    )

    geometry.crs = CRS.from_wkt(
        wkt
    )

    geometry.geod = Geod(
        geod_init
    )

    geometry.to_metric = Transformer.from_crs(
        4326,
        geometry.crs,
        always_xy=True,
    )

    geometry.to_lonlat = Transformer.from_crs(
        geometry.crs,
        4326,
        always_xy=True,
    )

    geometry.line_tree = STRtree(
        geometry.lines
    )

    return geometry


_local = threading.local()


def _initialize(
    geometry_type,
    snapshot,
    cfg,
    builder_type,
    event,
    process,
):
    if process:
        signal.signal(
            signal.SIGINT,
            signal.SIG_IGN,
        )

        logger.remove()

    _local.context = (
        cfg,
        _restore_geometry(
            geometry_type,
            snapshot,
        ),
        builder_type,
        event,
    )


def _run_task(task, context=None):
    from .geometry import InvalidScenarioError

    started = time.perf_counter()

    try:
        (
            cfg,
            geometry,
            builder_type,
            event,
        ) = (
            context
            if context is not None
            else _local.context
        )

        with logger.contextualize(
            mutation_worker=True,
            mutation_task=task.index,
        ):
            builder = builder_type(
                cfg,
                geometry,
                random.Random(task.seed),
            )

            builder.reference = task.parent
            builder.plan = task.plan
            builder.cancel_event = event

            cells = frozenset(
                builder.child(
                    task.parent,
                    task.donor,
                )
            )

            changed = len(
                task.parent ^ cells
            )

            if (
                len(cells) != len(task.parent)
                or not geometry.valid(cells)
                or (
                    changed
                    and not (
                        builder.minimum_change(task.parent)
                        <= changed
                        <= builder.change_limit(task.parent)
                    )
                )
            ):
                raise RuntimeError(
                    "Worker нарушил инварианты мутации"
                )

            stats = dict(
                builder.last_stats
            )

            stats.update(
                task_id=task.index,
                planned_azimuth_deg=(
                    task.plan.azimuth_deg
                ),
                planned_strip_position=(
                    task.plan.strip_position
                ),
                planned_strength=(
                    task.plan.strength
                ),
            )

            return MutationResult(
                index=task.index,
                seed=task.seed,
                cells=cells,
                seconds=(
                    time.perf_counter() - started
                ),
                stats=stats,
            )

    except Exception as error:
        return MutationResult(
            index=task.index,
            seed=task.seed,
            cells=task.parent,
            seconds=(
                time.perf_counter() - started
            ),
            stats={
                "status": "worker_error"
            },
            error=traceback.format_exc(),
            fatal=not isinstance(
                error,
                InvalidScenarioError,
            ),
        )


class ParallelMutationPool:
    def __init__(
        self,
        builder,
        backend=MUTATION_BACKEND,
        workers=MUTATION_WORKERS,
        *,
        rounds=MAX_GENERATION_ROUNDS,
        min_distance=MIN_PAIR_DISTANCE,
    ):
        if backend not in {
            "serial",
            "thread",
            "process",
        }:
            raise ValueError(
                "backend: serial | thread | process"
            )

        if any(
            type(value) is not int or value < 1
            for value in (
                workers,
                rounds,
            )
        ):
            raise ValueError(
                "workers и rounds должны быть целыми и > 0"
            )

        if not 0 <= min_distance <= 1:
            raise ValueError(
                "min_distance должно лежать в [0, 1]"
            )

        if builder.geometry.fixed_cell_count is None:
            raise ValueError(
                "Создавайте пул после builder.initial()"
            )

        self.backend = backend

        self.workers = (
            1
            if backend == "serial"
            else workers
        )

        self.rounds = rounds
        self.min_distance = min_distance

        self.last_results = []
        self.last_attempts = []
        self.last_stats = {}

        self._closed = False
        self._busy = False
        self.executor = None

        context = mp.get_context(
            "spawn"
        )

        self._event = (
            context.Event()
            if backend == "process"
            else threading.Event()
        )

        snapshot = _geometry_snapshot(
            builder.geometry
        )

        args = (
            type(builder.geometry),
            snapshot,
            builder.cfg,
            type(builder),
            self._event,
            backend == "process",
        )

        if backend == "serial":
            self._context = (
                builder.cfg,
                _restore_geometry(
                    type(builder.geometry),
                    snapshot,
                ),
                type(builder),
                self._event,
            )

        elif backend == "process":
            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize,
                initargs=args,
            )

        else:
            self.executor = ThreadPoolExecutor(
                max_workers=workers,
                initializer=_initialize,
                initargs=args,
            )

        logger.info(
            "ПУЛ МУТАЦИЙ | {} | workers={} | "
            "snapshot={:.1f} МиБ",
            backend,
            self.workers,
            len(snapshot) / 1024**2,
        )

    def close(self, *, abort=False):
        if self._closed:
            return

        self._closed = True

        if abort:
            self._event.set()

        executor = self.executor
        self.executor = None

        if executor is not None:
            terminate = getattr(
                executor,
                "terminate_workers",
                None,
            )

            if (
                abort
                and self.backend == "process"
                and terminate is not None
            ):
                terminate()

            else:
                executor.shutdown(
                    wait=not abort,
                    cancel_futures=True,
                )

    def __enter__(self):
        if self._closed:
            raise RuntimeError(
                "Пул закрыт"
            )

        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(
            abort=exc_type is not None
        )

    def _execute(self, tasks, label):
        results = []

        if self.backend == "serial":
            for task in tasks:
                result = _run_task(
                    task,
                    self._context,
                )

                if result.fatal:
                    raise RuntimeError(
                        f"seed={result.seed}:\n"
                        f"{result.error}"
                    )

                results.append(result)

            return results

        pending = {}
        iterator = iter(tasks)
        exhausted = False

        started = time.perf_counter()
        last_log = started

        try:
            while pending or not exhausted:
                while (
                    not exhausted
                    and len(pending)
                    < self.workers * MAX_PENDING_PER_WORKER
                ):
                    task = next(
                        iterator,
                        None,
                    )

                    if task is None:
                        exhausted = True

                    else:
                        future = self.executor.submit(
                            _run_task,
                            task,
                        )

                        pending[future] = task.index

                if not pending:
                    break

                ready, _ = wait(
                    pending,
                    timeout=1,
                    return_when=FIRST_COMPLETED,
                )

                for future in ready:
                    pending.pop(future)

                    result = future.result()

                    if result.fatal:
                        raise RuntimeError(
                            f"seed={result.seed}:\n"
                            f"{result.error}"
                        )

                    results.append(result)

                now = time.perf_counter()

                if (
                    now - last_log
                    >= PROGRESS_INTERVAL_S
                ):
                    logger.info(
                        "{} | готово {}/{} | {:.1f} с",
                        label,
                        len(results),
                        len(tasks),
                        now - started,
                    )

                    last_log = now

        except BaseException:
            for future in pending:
                future.cancel()

            raise

        return sorted(
            results,
            key=lambda result: result.index,
        )

    def generate(
        self,
        parent,
        donors,
        seeds,
        label="Мутации",
    ):
        if self._closed or self._busy:
            raise RuntimeError(
                "Пул закрыт или уже выполняет generate()"
            )

        self.last_results = []
        self.last_attempts = []
        self.last_stats = {}

        parent = frozenset(parent)
        donors = list(donors)
        seeds = list(seeds)

        if (
            len(donors) != len(seeds)
            or any(
                type(seed) is not int
                for seed in seeds
            )
        ):
            raise ValueError(
                "Нужны одинаковые количества доноров "
                "и целочисленных seed"
            )

        wanted = len(seeds)

        if not wanted:
            return []

        started = time.perf_counter()

        unique = {}
        parent_returns = 0
        duplicates = 0
        selected = []

        self._busy = True

        try:
            for round_index in range(self.rounds):
                tasks = [
                    make_task(
                        round_index * wanted + index,
                        round_index,
                        parent,
                        frozenset(donors[index]),
                        seeds[index],
                        seeds[0] % DIRECTION_SECTORS,
                    )
                    for index in range(wanted)
                ]

                results = self._execute(
                    tasks,
                    (
                        f"{label}: партия "
                        f"{round_index + 1}"
                    ),
                )

                self.last_attempts.extend(
                    results
                )

                for result in results:
                    if result.error:
                        logger.warning(
                            "{} | seed={} | {}",
                            label,
                            result.seed,
                            result.error,
                        )

                    if result.cells == parent:
                        parent_returns += 1

                    elif result.cells in unique:
                        duplicates += 1

                    else:
                        unique[result.cells] = result

                selected = select_diverse(
                    list(unique.values()),
                    wanted,
                    self.min_distance,
                )

                logger.info(
                    "{} | партия={} | уникальных={} | "
                    "достаточно разных={}/{} | возвратов={}",
                    label,
                    round_index + 1,
                    len(unique),
                    len(selected),
                    wanted,
                    parent_returns,
                )

                if len(selected) == wanted:
                    break

            # Ослабляется только порог попарного разнообразия.
            # Геометрические ограничения остаются прежними.
            effective = self.min_distance

            if len(selected) < wanted:
                for effective in (
                    self.min_distance / 2,
                    0.0,
                ):
                    selected = select_diverse(
                        list(unique.values()),
                        wanted,
                        effective,
                    )

                    if len(selected) == wanted:
                        break

            distance = min(
                (
                    cell_distance(
                        first.cells,
                        second.cells,
                    )
                    for index, first in enumerate(selected)
                    for second in selected[index + 1:]
                ),
                default=None,
            )

            rejections = Counter()

            for row in self.last_attempts:
                rejections.update(
                    row.stats.get(
                        "rejections",
                        {},
                    )
                )

            self.last_stats = dict(
                requested=wanted,
                selected=len(selected),
                partial=len(selected) < wanted,

                attempts=len(self.last_attempts),
                unique=len(unique),
                parent_returns=parent_returns,
                duplicates=duplicates,

                effective_threshold=effective,
                actual_min_distance=distance,

                max_changed_seen=max(
                    (
                        row.stats.get(
                            "best_changed_seen",
                            0,
                        )
                        for row in self.last_attempts
                    ),
                    default=0,
                ),

                rejections=dict(rejections),
                seconds=(
                    time.perf_counter() - started
                ),
            )

            self.last_results = [
                replace(
                    result,
                    index=index,
                    stats={
                        **result.stats,
                        "generation_sampling": dict(
                            self.last_stats
                        ),
                    },
                )
                for index, result in enumerate(selected)
            ]

            if len(selected) < wanted:
                logger.warning(
                    "{} | неполное поколение: {}/{} | "
                    "найденные формы передаются на оценку",
                    label,
                    len(selected),
                    wanted,
                )

            if effective < self.min_distance:
                logger.warning(
                    "{} | порог разнообразия: {} -> {}",
                    label,
                    self.min_distance,
                    effective,
                )

            logger.info(
                "{} | диагностика: {}",
                label,
                self.last_stats,
            )

            return [
                result.cells
                for result in self.last_results
            ]

        except BaseException:
            self.close(abort=True)
            raise

        finally:
            self._busy = False