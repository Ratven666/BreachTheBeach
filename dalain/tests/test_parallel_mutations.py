import pytest
from dalain.anchored import GeneticBuilder
from dalain.parallel_mutations import ParallelMutationPool


class BrokenBuilder(GeneticBuilder):
    def child(self, first, second):
        raise RuntimeError("intentional worker failure")


def test_spawn_thread_serial_same_shapes(anchored_builder):
    b = anchored_builder
    parent = b.initial()
    seeds = [41, 42, 43, 44]
    results = {}
    for backend in ("serial", "thread", "process"):
        with ParallelMutationPool(b, backend=backend, workers=2, tasks_per_batch=2) as pool:
            results[backend] = pool.generate(parent, [parent] * len(seeds), seeds)
            assert [r.seed for r in pool.last_results] == seeds
            assert [r.index for r in pool.last_results] == list(range(len(seeds)))
            assert not any(r.error for r in pool.last_results)
            assert all(b.geometry.valid(c) for c in results[backend])
            from dalain.optimize import validate_population_candidate
            for candidate in results[backend]:
                validate_population_candidate(b.geometry, b, parent, candidate)
            assert pool.generate(parent, [], []) == []
            assert pool.last_results == []
            again = pool.generate(parent, [parent] * len(seeds), seeds)
            assert again == results[backend]
    assert results["serial"] == results["thread"] == results["process"]
    assert any(c != parent for c in results["process"])


def test_two_serial_pools_do_not_overwrite_state(anchored_builder):
    b = anchored_builder
    parent = b.initial()
    broken = BrokenBuilder(b.cfg, b.geometry, b.rng)
    broken.reference = parent
    with ParallelMutationPool(b, backend="serial") as first:
        with ParallelMutationPool(broken, backend="serial"):
            assert len(first.generate(parent, [parent], [5])) == 1


@pytest.mark.parametrize("backend", ["serial", "process"])
def test_worker_failure_is_fatal_and_closes_pool(anchored_builder, backend):
    b = anchored_builder
    parent = b.initial()
    broken = BrokenBuilder(b.cfg, b.geometry, b.rng)
    broken.reference = parent
    pool = ParallelMutationPool(broken, backend=backend, workers=2)
    with pytest.raises(RuntimeError, match="intentional worker failure"):
        pool.generate(parent, [parent], [1])
    with pytest.raises(RuntimeError, match="закрыт"):
        pool.generate(parent, [], [])
    assert pool.last_results == []
    pool.close()


def test_pool_lifecycle(anchored_builder):
    b = anchored_builder
    with pytest.raises(ValueError, match="initial"):
        ParallelMutationPool(b)
    parent = b.initial()
    pool = ParallelMutationPool(b, backend="serial")
    with pytest.raises(ValueError):
        pool.generate(parent, [parent], [])
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="закрыт"):
        pool.generate(parent, [], [])


class WaitingBuilder(GeneticBuilder):
    def child(self, first, second):
        from concurrent.futures import CancelledError
        if not self.cancel_event.wait(5):
            raise RuntimeError("Отмена не дошла до worker")
        raise CancelledError("worker остановлен")


def test_process_batch_timeout(anchored_builder):
    import time
    b = anchored_builder
    parent = b.initial()
    waiting = WaitingBuilder(b.cfg, b.geometry, b.rng)
    waiting.reference = parent
    pool = ParallelMutationPool(waiting, backend="process", workers=2, batch_timeout_s=0.1)
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        pool.generate(parent, [parent], [1])
    assert time.perf_counter() - started < 4
    assert pool.last_results == []
    assert pool._cancel_event.is_set()


def test_operator_cooperative_cancellation(anchored_builder):
    import threading
    from concurrent.futures import CancelledError
    b = anchored_builder
    parent = b.initial()
    b.cancel_event = threading.Event()
    b.cancel_event.set()
    with pytest.raises(CancelledError):
        b.mutate(parent)


def test_serial_geometry_snapshot_is_isolated(anchored_builder):
    b = anchored_builder
    b.initial()
    with ParallelMutationPool(b, backend="serial") as pool:
        assert pool._state.geometry is not b.geometry
        assert pool._state.geometry.cells == b.geometry.cells
        assert pool._state.geometry.to_lonlat is not b.geometry.to_lonlat
