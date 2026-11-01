import random
from dalain.anchored import ANCHOR_CELL


def test_dynamic_front_and_count(anchored_builder):
    b = anchored_builder
    parent = b.initial()
    successes = 0
    for seed in range(6):
        b.rng = random.Random(seed)
        child = b.mutate(parent)
        current = set(parent)
        previous_changed = 0
        for i in range(0, len(b.last_trace), 2):
            action, removed = b.last_trace[i]
            assert action == "remove"
            removable, _ = b._sea_frontiers(current)
            assert removed in removable and removed != ANCHOR_CELL
            current.remove(removed)
            action, added = b.last_trace[i + 1]
            assert action == "add" and added != removed
            _, addable = b._sea_frontiers(current)
            assert added in addable
            current.add(added)
            assert len(current) == len(parent)
            changed = len(parent ^ frozenset(current))
            assert changed >= previous_changed
            previous_changed = changed
        assert frozenset(current) == child
        assert len(child) == len(parent)
        assert b.geometry.valid(child)
        if child != parent:
            successes += 1
            assert b.minimum_change(parent) <= len(parent ^ child) <= b.change_limit(parent)
        else:
            assert b.last_stats["status"] == "no_feasible_mutation"
        parent = child
    assert successes > 0
