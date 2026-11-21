"""Постоянный пул оценки CWEF. Базу контрольных точек открывает только родитель."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from multiprocessing import get_context
from multiprocessing.util import Finalize
import signal
import sys

from loguru import logger

from dalain import sequential as seq


_evaluator = None


def _initialize(grid_cfg, cfg, default_target, target, detailed_bathymetry):
    global _evaluator
    # Ctrl+C обрабатывает родитель. У каждого процесса свои SQLite/растр/кэши.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger.remove()
    logger.add(
        sys.stderr, level="WARNING", diagnose=False,
        format="{time:HH:mm:ss} | CWEF {process.id} | {level} | {message}",
        filter=lambda record: detailed_bathymetry or not (
            record["name"] == "dalain.physics"
            and record["message"].startswith("НЕТ БАТИМЕТРИИ |")
        ),
    )
    geometry = seq.SequenceGeometry(grid_cfg, default_target)
    geometry.target = target
    geometry.cfg = cfg
    _evaluator = seq.Evaluator(cfg, geometry)
    Finalize(None, _evaluator.close, exitpriority=10)


def _evaluate(current, label):
    # Не сериализуем Shapely-геометрию: ей не нужно покидать рабочий процесс.
    return _evaluator.evaluate(current, label).metrics


class EvaluationPool:
    def __init__(self, workers, grid_cfg, cfg, default_target, target, detailed_bathymetry):
        self.workers = workers
        self.pending = {}
        self.executor = None
        if workers > 1:
            self.executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize,
                initargs=(grid_cfg, cfg, default_target, target, detailed_bathymetry),
            )

    def submit(self, key, current, label):
        if key not in self.pending:
            self.pending[key] = self.executor.submit(_evaluate, current, label)

    def take(self, key, report, interval):
        future = self.pending.pop(key, None)
        if future is None:
            return None
        while not future.done():
            done, _ = wait((future,), timeout=interval, return_when=FIRST_COMPLETED)
            if not done:
                report(1 + sum(not item.done() for item in self.pending.values()))
        return future.result()

    def discard(self):
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()

    def close(self):
        if self.executor is not None:
            self.discard()
            self.executor.shutdown(wait=True, cancel_futures=True)

    def batches(self, proposals, size, prefetch):
        """Порядок выдачи не зависит от порядка завершения процессов."""
        if self.executor is None:
            yield from proposals
            return
        while batch := list(islice(proposals, max(1, min(self.workers, size())))):
            self.discard()
            prefetch(batch)
            yield from batch
