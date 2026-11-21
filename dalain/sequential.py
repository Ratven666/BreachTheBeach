"""Поэтапное строительство фиксированного контура: python -m dalain.sequential."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import random
import sys

from loguru import logger
from shapely import STRtree, coverage_union_all, transform, union_all
from shapely.geometry import Point, mapping, shape

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dalain.anchored import ANCHOR_CELL, AnchoredGeometry, GeneticBuilder
from dalain.bathymetry import prepare_bathymetry
from dalain.config import PROJECT_ROOT, Config
from dalain.geometry import Geometry, InvalidScenarioError, is_connected, neighbours
from dalain.logging_utils import configure_logging
from dalain.optimize import (
    Evaluator, Recorder, checked_score, mean_normal_cwef, normalize_paths, write_json,
)


@dataclass(frozen=True)
class SequenceConfig:
    target_path: Path | None
    stage_count: int
    stage_cell_counts: tuple[int, ...] | None = None
    stage_weights: tuple[float, ...] | None = None
    immigrant_probability: float = 0.20
    make_animation: bool = True
    stage_size_tolerance: float = 0.10
    save_search_solutions: bool = False


class SequenceGeometry(AnchoredGeometry):
    def __init__(self, cfg, build_default_target):
        self.target = None
        if build_default_target:
            super().__init__(cfg)
        else:
            Geometry.__init__(self, cfg)
            self.anchor_cell = ANCHOR_CELL
            self.anchor_point = Point(0, 0)

    def valid(self, cells):
        if self.target is None:
            return super().valid(cells)
        return (
            cells <= self.target
            and is_connected(cells)
            and (self.cfg.connection == "island" or any(self.units[c] == 1 for c in cells))
        )

    def scenario(self, cells):
        # Здесь штатный Geometry.scenario сам проверяет полости, водоёмы и нормали.
        return Geometry.scenario(self, cells)

    def _fill(self, cells):
        if not cells:
            return super()._fill(cells)
        return self._q(coverage_union_all([self.cells[c] for c in cells]))


def read_target(path, geometry):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["type"] == "FeatureCollection":
        features = data["features"]
    elif data["type"] == "Feature":
        features = [data]
    else:
        features = [{"geometry": data}]

    groups = {"reclamation": [], "cell": [], None: []}
    for feature in features:
        kind = (feature.get("properties") or {}).get("kind")
        obj = feature.get("geometry")
        if kind in groups and obj and obj["type"] in {"Polygon", "MultiPolygon"}:
            groups[kind].append(shape(obj))

    parts = groups["reclamation"] or groups["cell"] or groups[None]
    if not parts or any(p.is_empty or not p.is_valid for p in parts):
        raise ValueError(f"{path}: нет корректного полигона намыва")

    contour = geometry._q(union_all([
        geometry._q(transform(p, geometry.to_metric.transform, interleaved=False))
        for p in parts
    ]))
    keys = tuple(geometry.cells)
    squares = [geometry.cells[c] for c in keys]
    target = frozenset(
        keys[i] for i in STRtree(squares).query(contour, predicate="covers")
    )
    if not target or not contour.equals(geometry._fill(target)):
        raise ValueError(
            f"{path}: контур не совпадает с допустимыми полными клетками. "
            "Проверьте WGS84, центр, сторону клетки, радиус, берег и режим connection."
        )
    return target


def stage_parameters(options, cell_count):
    n = options.stage_count
    if type(n) is not int or not 1 <= n <= cell_count:
        raise ValueError("stage_count должно быть от 1 до количества клеток")
    counts = options.stage_cell_counts
    if counts is None:
        q, r = divmod(cell_count, n)
        counts = tuple(q + (i < r) for i in range(n))
    if (len(counts) != n or any(type(k) is not int or k < 1 for k in counts)
            or sum(counts) != cell_count):
        raise ValueError(f"stage_cell_counts: нужны {n} положительных целых с суммой {cell_count}")

    weights = options.stage_weights if options.stage_weights is not None else (1.0,) * n
    if len(weights) != n or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError(f"stage_weights: нужны {n} конечных положительных весов")
    scale = max(weights)
    weights = tuple(w / scale for w in weights)
    total = math.fsum(weights)
    return counts, tuple(w / total for w in weights)


def stage_bounds(counts, tolerance):
    if not 0 <= tolerance < 1:
        raise ValueError("stage_size_tolerance должно лежать в [0, 1)")
    return tuple(
        (max(1, n - math.floor(n * tolerance)), n + math.floor(n * tolerance))
        for n in counts
    )


def connected_cut(remaining, order, lo, hi, preferred):
    """Ближайший допустимый срез; при необходимости проверяем весь диапазон за один проход."""
    rest = remaining.difference(order[:preferred])
    if is_connected(rest):
        return preferred
    if lo == hi:
        return None

    # При движении назад клетки только добавляются: компоненты поддерживает DSU.
    parent, size = {}, {}
    components = 0

    def root(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    def add(cell):
        nonlocal components
        parent[cell], size[cell] = cell, 1
        components += 1
        for other in neighbours(cell):
            if other not in parent:
                continue
            a, b = root(cell), root(other)
            if a == b:
                continue
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            components -= 1

    for cell in remaining.difference(order[:hi]):
        add(cell)
    best = None
    for cut in range(hi, lo - 1, -1):
        if components == 1 and (best is None or
                (abs(cut - preferred), cut) < (abs(best - preferred), best)):
            best = cut
        if cut > lo:
            add(order[cut - 1])
    return best


def decode(target, starts, counts, genes, bounds=None):
    """Гены: точка притяжения, направление и предпочтительный сдвиг размера."""
    if bounds is None:
        bounds = stage_bounds(counts, 0.0)
    suffix_min = [0] * (len(counts) + 1)
    suffix_max = suffix_min.copy()
    for i in range(len(counts) - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + bounds[i][0]
        suffix_max[i] = suffix_max[i + 1] + bounds[i][1]

    remaining = set(target)
    front = set(starts)
    stages = []
    for index, count in enumerate(counts):
        low, high = bounds[index]
        lo = max(low, len(remaining) - suffix_max[index + 1])
        hi = min(high, len(remaining) - suffix_min[index + 1])
        if not front or lo > hi:
            return None
        if index == len(counts) - 1:
            if index == 0 and not is_connected(remaining):
                return None
            return (*stages, frozenset(remaining))

        (x, y), angle, bias = genes[index]
        preferred = min(hi, max(lo, count + round(bias * (high - count))))
        dx, dy = math.cos(angle), math.sin(angle)
        seed = min(front, key=lambda c: ((c[0] - x)**2 + (c[1] - y)**2, c))
        order = []
        queued = {seed}
        heap = [(0.0, seed)]
        while heap and len(order) < hi:
            _, cell = heappop(heap)
            order.append(cell)
            for c in neighbours(cell):
                if c in remaining and c not in queued:
                    queued.add(c)
                    heappush(heap, (c[0] * dx + c[1] * dy, c))
        hi = min(hi, len(order))
        if hi < lo:
            return None
        cut = connected_cut(remaining, order, lo, hi, min(preferred, hi))
        if cut is None:
            return None

        stage = frozenset(order[:cut])
        stages.append(stage)
        remaining.difference_update(stage)
        if index == 0:
            front.clear()  # Береговые старты больше не заменяют касание намыва.
        front.difference_update(stage)
        front.update(c for cell in stage for c in neighbours(cell) if c in remaining)
    raise AssertionError("Нет последней заходки")


def search(cfg, target, starts, counts, score_plan, rng, immigrant_probability, bounds=None):
    """Турнирная селекция, элита, кроссовер и мутация с декодером связности."""
    cells = sorted(target)
    n_genes = len(counts) - 1

    def gene():
        return rng.choice(cells), rng.uniform(0.0, math.tau), rng.uniform(-1.0, 1.0)

    def random_genome():
        return tuple(gene() for _ in range(n_genes))

    def child(population):
        def tournament():
            return min(
                rng.choices(population, k=cfg.tournament_size),
                key=lambda member: member[0],
            )[1]

        first = tournament()
        genes = list(first)
        if n_genes and rng.random() < cfg.crossover_probability:
            second = tournament()
            genes = [rng.choice((a, b)) for a, b in zip(first, second)]
        if n_genes and rng.random() < cfg.mutation_probability:
            size = max(1, math.ceil(n_genes * cfg.mutation_fraction))
            for i in rng.sample(range(n_genes), size):
                genes[i] = gene()
        return tuple(genes)

    population, history = [], []
    seen = set()
    for epoch in range(cfg.epochs):
        for generation in range(cfg.generations):
            next_pop = population[:cfg.elite_count]
            new = 0
            for _ in range(cfg.construction_attempts):
                if len(next_pop) >= cfg.population_size:
                    break
                genome = (
                    child(population)
                    if population and rng.random() >= immigrant_probability
                    else random_genome()
                )
                plan = decode(target, starts, counts, genome, bounds)
                if plan is None or plan in seen:
                    continue
                seen.add(plan)
                score = score_plan(plan, epoch, generation, new)
                if score is not None:
                    next_pop.append((score, genome, plan))
                    new += 1
                    if not n_genes:
                        return plan, [dict(epoch=epoch, generation=generation, score=score, new_plans=1)]
            if not next_pop:
                raise RuntimeError(
                    "Не найдено допустимое разбиение. Измените размеры/число заходок "
                    "или увеличьте stage_size_tolerance либо construction_attempts."
                )
            population = sorted(next_pop, key=lambda member: member[0])
            score = population[0][0]
            history.append(dict(epoch=epoch, generation=generation, score=score, new_plans=new))
            logger.info("Заходки | эпоха={} поколение={} новых={} лучший score={:.6f}",
                        epoch + 1, generation + 1, new, score)
            if not new:
                return population[0][2], history
    return population[0][2], history


def run_sequential(cfg, options, objective=mean_normal_cwef):
    cfg = normalize_paths(cfg)
    if not 0 <= options.immigrant_probability <= 1:
        raise ValueError("immigrant_probability должно лежать в [0, 1]")
    path = options.target_path
    if path is not None:
        path = (PROJECT_ROOT / Path(path).expanduser()).resolve()
        # Для построения сетки внешнего контура не нужна заданная вручную площадь.
        cfg = replace(cfg, target_area_m2=2 * cfg.cell_side_m**2)
    cfg.validate()
    wal = Path(f"{cfg.weather_path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f"Погодная БД содержит непогашенный WAL: {wal}")

    directory = cfg.output_directory / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True)
    configure_logging(cfg, directory)

    geometry = SequenceGeometry(cfg, build_default_target=path is None)
    if path is None:
        target = GeneticBuilder(cfg, geometry, random.Random(cfg.random_seed)).initial()
    else:
        target = read_target(path, geometry)
    geometry.target = target
    area = sum(geometry.units[c] for c in target) * cfg.cell_side_m**2 / 2
    starts = frozenset(c for c in target if cfg.connection == "island" or geometry.units[c] == 1)
    if len(target) < 2 or not starts or not geometry.valid(target):
        raise ValueError("Целевой намыв должен быть связан по рёбрам и соответствовать connection")
    counts, weights = stage_parameters(options, len(target))
    bounds = stage_bounds(counts, options.stage_size_tolerance)
    cfg = prepare_bathymetry(replace(cfg, target_area_m2=area))
    geometry.cfg = cfg
    name = getattr(objective, "__name__", type(objective).__name__)
    write_json(directory / "config.json", {
        **asdict(cfg), "sequence": {**asdict(options), "target_path": path},
        "stage_cell_counts_resolved": counts, "stage_cell_count_bounds": bounds,
        "stage_weights_normalized": weights,
        "objective": name,
    })

    def feature(polygon, **properties):
        return dict(type="Feature", geometry=mapping(geometry.wgs84(polygon)), properties=properties)

    with ExitStack() as stack:
        evaluator = Evaluator(cfg, geometry)
        stack.callback(evaluator.close)
        recorder = Recorder(cfg, geometry, directory, name)
        stack.callback(recorder.close)
        baseline = evaluator.calculate(frozenset(), "Исходный берег")
        evaluator.baseline = baseline
        base_score = checked_score(objective, baseline, baseline)
        _, base_id = recorder.save(baseline, base_score, -1, -1, -1)
        target_result = evaluator.evaluate(target, "Целевой контур")
        write_json(directory / "target.geojson", dict(type="FeatureCollection", features=[
            feature(target_result.scenario.fill, kind="reclamation", area_m2=area),
        ]))

        scores = {target: checked_score(objective, target_result, baseline)}
        solution_ids = {frozenset(): base_id}
        invalid = set()

        def score_plan(plan, epoch, generation, individual):
            previous = frozenset()
            values = []
            for step, part in enumerate(plan, 1):
                current = previous | part
                if current in invalid:
                    return None
                result = None
                if current not in scores:
                    try:
                        result = (
                            target_result if current == target else
                            evaluator.evaluate(current, f"Кандидат: заходка {step}")
                        )
                    except InvalidScenarioError as exc:
                        invalid.add(current)
                        logger.debug("Заходка {} отклонена: {}", step, exc)
                        return None
                    scores[current] = checked_score(objective, result, baseline)
                if options.save_search_solutions and current not in solution_ids:
                    if result is None:
                        result = target_result if current == target else evaluator.evaluate(
                            current, f"Кандидат: заходка {step}"
                        )
                    _, solution_id = recorder.save(
                        result, scores[current], epoch, generation, individual,
                        parent_cells=previous, parent_solution_id=solution_ids[previous],
                    )
                    solution_ids[current] = solution_id
                values.append(scores[current])
                previous = current
            return math.fsum(w * v for w, v in zip(weights, values))

        plan, history = search(
            cfg, target, starts, counts, score_plan,
            random.Random(cfg.random_seed), options.immigrant_probability, bounds,
        )
        write_json(directory / "evolution.json", history)
        previous, parent_id = frozenset(), base_id
        stages, features, curve = [], [], []
        color_max = max(
            row["mean_CWEF_Wm"]
            for result in (baseline, target_result) for row in result.metrics
        )
        for step, part in enumerate(plan, 1):
            current = previous | part
            result = (
                target_result if current == target else
                evaluator.evaluate(current, f"Принятая заходка {step}")
            )
            value = scores[current]
            saved, parent_id = recorder.save(
                result, value, -1, step - 1, 0, parent_cells=previous, parent_solution_id=parent_id,
            )
            recorder.accept(saved)
            info = dict(
                stage=step, solution_id=parent_id, cell_count=len(part),
                nominal_cell_count=counts[step - 1], cell_count_bounds=bounds[step - 1],
                area_m2=sum(geometry.units[c] for c in part) * cfg.cell_side_m**2 / 2,
                water_area_m2=math.fsum(geometry.water_areas[c] for c in part),
                cumulative_area_m2=result.scenario.credited_area_m2,
                score=value, mean_CWEF_Wm=result.mean_cwef,
            )
            stages.append(info)
            features.append(feature(geometry._fill(part), kind="stage", **info))
            curve.append(value)
            if options.make_animation:
                recorder.frame(result, 0, step - 1, curve, color_max, parent_cells=previous)
            previous = current
        if options.make_animation:
            recorder.animate()
        write_json(directory / "stages.geojson", dict(type="FeatureCollection", features=features))
        write_json(directory / "summary.json", dict(
            target_area_m2=area, score=history[-1]["score"], stages=stages,
            objective=name, final_solution_id=parent_id,
        ))

    logger.success("Последовательность намыва сохранена: {}", directory)
    logger.complete()
    return directory


def main():
    cfg = Config(
        coastline_path=PROJECT_ROOT / "Kaliningrad/data/coastline/kaliningrad_OFFSET_merged.geojson",
        weather_path=PROJECT_ROOT / "Kaliningrad/data/db/weather_compact.db",
        land_path=None,
        bathymetry_path=None,
        bathymetry_auto_download=True,
        bathymetry_cache_directory=PROJECT_ROOT / "dalain/data/bathymetry",
        output_directory=PROJECT_ROOT / "dalain/output/sequential",
        center_lon=19.95, center_lat=54.74,
        target_area_m2=1_000_000.0,  # Используется только при target_path=None.
        cell_side_m=20.0, search_radius_m=1_000.0, evaluation_radius_m=1_250.0,
        connection="attached", sea_side="right",
        start_date="2025-01-01", end_date="2025-12-31",
        direction_count=36, point_spacing_m=50.0, tangent_delta_m=20.0,
        epochs=2, generations=12, population_size=12, construction_attempts=300,
        random_seed=42,
    )
    options = SequenceConfig(
        target_path=None,  # Или Path("dalain/output/<запуск>/solutions/solution_00000001.geojson").
        stage_count=5,
        stage_cell_counts=None,
        stage_weights=None,
        immigrant_probability=0.20,
        make_animation=False,
        stage_size_tolerance=0.10,
        save_search_solutions=False,
    )
    run_sequential(cfg, options, objective=mean_normal_cwef)


if __name__ == "__main__":
    main()
