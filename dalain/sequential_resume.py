"""Возобновляемая оптимизация: python -m dalain.sequential_resume."""
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import math
import random
import sqlite3
import sys
import time
import zlib

from loguru import logger
from shapely.geometry import mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dalain import sequential as seq
from dalain.optimize import Evaluation, json_default
from dalain.sequential_animation import animation


@dataclass(frozen=True)
class RunOptions:
    directory: Path
    mode: str = "ga"                 # ga | greedy
    greedy_trials: int = 48         # На одну заходку.
    greedy_tail_tolerance: float = 0.30
    objective_id: str = "mean_normal_cwef:v1"
    make_animation: bool = True
    render_only: bool = False


def dump(obj):
    return json.dumps(obj, default=json_default, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def pack(obj):
    return zlib.compress(dump(obj), level=1)


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def digest(obj):
    return sha256(dump(obj)).hexdigest()


def cells(rows):
    return frozenset(map(tuple, rows))


def tuples(obj):
    return tuple(map(tuples, obj)) if isinstance(obj, list) else obj


def path_stamp(path):
    if path is None:
        return None
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


class Store:
    def __init__(self, db, manifest):
        self.db = db
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, val BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL, metrics BLOB);
            CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS events (
                step INTEGER, attempt INTEGER, sub INTEGER, payload BLOB NOT NULL,
                PRIMARY KEY(step, attempt, sub)
            ) WITHOUT ROWID;
        """)
        value = self.read("manifest")
        if value is not None and value != manifest:
            raise ValueError("Входные файлы или настройки изменились. Используйте другой каталог.")
        if value is None:
            with db:
                self.put("manifest", manifest)

    def read(self, key):
        row = self.db.execute("SELECT val FROM meta WHERE key=?", (key,)).fetchone()
        return unpack(row[0]) if row else None

    def put(self, key, val):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, pack(val)))

    def commit_state(self, state, seen=(), checkpoint=None):
        with self.db:
            self.put("state", state)
            self.db.executemany("INSERT OR IGNORE INTO seen VALUES (?)", ((k,) for k in seen))
            if checkpoint is not None:
                step, attempt, fields = checkpoint
                self._insert_event(step, attempt, 0, "checkpoint", **fields)

    def was_seen(self, key):
        return self.db.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone() is not None

    def get_score(self, key):
        return self.db.execute("SELECT score FROM scores WHERE key=?", (key,)).fetchone()

    def get_metrics(self, key):
        row = self.db.execute("SELECT metrics FROM scores WHERE key=?", (key,)).fetchone()
        return unpack(row[0]) if row and row[0] is not None else None

    def add_score(self, key, score, metrics=None):
        with self.db:
            self.db.execute("INSERT INTO scores VALUES (?, ?, ?)",
                            (key, score, pack(metrics) if metrics is not None else None))

    def event(self, step, attempt, sub, status, **fields):
        with self.db:
            self._insert_event(step, attempt, sub, status, **fields)

    def _insert_event(self, step, attempt, sub, status, **fields):
        self.db.execute("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)",
                        (step, attempt, sub, pack(dict(
                            step=step, attempt=attempt if status != "checkpoint" else None,
                            status=status, **fields,
                        ))))

    def events(self):
        for (blob,) in self.db.execute("SELECT payload FROM events ORDER BY step, attempt, sub"):
            yield unpack(blob)


class Scorer:
    def __init__(self, store, evaluator, geometry, objective):
        self.store, self.evaluator, self.geometry = store, evaluator, geometry
        self.objective = objective
        self.hits = self.calculated = 0
        empty = frozenset()
        key = digest(empty)
        metrics = store.get_metrics(key)
        if metrics is None:
            self.baseline = evaluator.calculate(empty, "Исходный берег")
            value = seq.checked_score(objective, self.baseline, self.baseline)
            store.add_score(key, value, self.baseline.metrics)
        else:
            self.baseline = Evaluation(geometry.scenario(empty), metrics)
        evaluator.baseline = self.baseline

    def score(self, current):
        key = digest(current)
        row = self.store.get_score(key)
        if row is not None:
            self.hits += 1
            return row[0]  # NULL означает геометрически недопустимый вариант.
        try:
            result = self.evaluator.evaluate(current, f"CWEF {key[:8]} | клеток={len(current)}")
        except seq.InvalidScenarioError as exc:
            logger.debug("Геометрия {} отклонена: {}", key[:8], exc)
            self.store.add_score(key, None)
            return None
        value = seq.checked_score(self.objective, result, self.baseline)
        self.store.add_score(key, value, result.metrics)
        self.calculated += 1
        return value


def cumulative(plan):
    built = frozenset()
    for part in plan:
        built |= part
        yield built


def random_gene(points, rng):
    return rng.choice(points), rng.uniform(0, math.tau), rng.uniform(-1, 1)


def evolve(cfg, options, target, starts, counts, weights, scorer, store, state, publish):
    max_gen = cfg.epochs * cfg.generations
    bounds = seq.stage_bounds(counts, options.stage_size_tolerance)
    points, n = sorted(target), len(counts) - 1
    rng = random.Random()
    rng.setstate(tuples(state["rng"]))
    while not state["done"]:
        gen = state["step"]
        parents = state["population"]
        population = parents[:cfg.elite_count]
        new = set()
        best, best_score = state["plan"], state["best_score"]
        accepted = 0
        last = -math.inf
        for attempt in range(1, cfg.construction_attempts + 1):
            if len(population) >= cfg.population_size:
                break
            if time.monotonic() - last >= cfg.progress_interval_s:
                logger.info("ГА {}/{} | попытка {}/{} | новых={}/{} | CWEF={} | кеш={}",
                            gen + 1, max_gen, attempt, cfg.construction_attempts, accepted,
                            cfg.population_size - min(len(parents), cfg.elite_count),
                            scorer.calculated, scorer.hits)
                last = time.monotonic()
            if parents and rng.random() >= options.immigrant_probability:
                def tournament():
                    return min(rng.choices(parents, k=cfg.tournament_size), key=lambda p: p[0])[1]
                genes = list(tournament())
                if n and rng.random() < cfg.crossover_probability:
                    genes = [rng.choice((a, b)) for a, b in zip(genes, tournament())]
                if n and rng.random() < cfg.mutation_probability:
                    for i in rng.sample(range(n), math.ceil(n * cfg.mutation_fraction)):
                        genes[i] = random_gene(points, rng)
            else:
                genes = [random_gene(points, rng) for _ in range(n)]
            plan = seq.decode(target, starts, counts, genes, bounds)
            if plan is None:
                store.event(gen + 1, attempt, 0, "decode",
                            seed=genes[0][0] if genes else None,
                            note="Decoder found no complete connected plan")
                continue
            key = digest(plan)
            if key in new or store.was_seen(key):
                store.event(gen + 1, attempt, 0, "duplicate", plan=plan)
                continue
            new.add(key)
            values = []
            for i, built in enumerate(cumulative(plan), 1):
                value = scorer.score(built)
                store.event(gen + 1, attempt, i, "trial" if value is not None else "invalid",
                            plan=plan[:i], score=value, active=i,
                            note=f"Accumulated state after stage {i}/{len(plan)}")
                if value is None:
                    break
                values.append(value)
            if len(values) != len(counts):
                continue
            score = math.fsum(a * b for a, b in zip(weights, values))
            accepted += 1
            population.append([score, genes])
            improved = best_score is None or score < best_score
            if improved:
                best, best_score = plan, score
            store.event(gen + 1, attempt, len(plan) + 1, "leader" if improved else "worse",
                        plan=plan, score=score, best=best_score, score_kind="Weighted plan score",
                        note="Compared with the best complete plan so far")
            if n == 0:
                break
        if not population:
            raise RuntimeError("Нет допустимого плана. Проверьте размеры этапов и допуск.")
        population.sort(key=lambda p: p[0])
        state.update(
            step=gen + 1, population=population, plan=best, best_score=best_score,
            rng=rng.getstate(), done=(not accepted or gen + 1 == max_gen or n == 0),
        )
        state["history"].append(dict(generation=gen + 1, new_plans=accepted, score=best_score))
        store.commit_state(state, new, (gen + 1, cfg.construction_attempts + 1, dict(
            plan=best, score=best_score, score_kind="Weighted plan score",
            note="Best complete plan so far",
        )))
        publish(state)
        logger.info("ГА: сохранено поколение {}/{} | score={:.8g}", gen + 1, max_gen, best_score)


def greedy(cfg, options, run, target, starts, counts, scorer, store, state, publish):
    rng = random.Random()
    rng.setstate(tuples(state["rng"]))
    plan = [cells(part) for part in state["plan"]]
    built = frozenset().union(*plan)
    while not state["done"]:
        i = len(plan)
        remaining = target - built
        front = ({c for b in built for c in seq.neighbours(b) if c in remaining}
                 if built else starts)
        future = len(counts) - i - 1
        best, best_score = None, None
        if future and len(remaining) > future and front:
            nominal = max(1, min(counts[i], len(remaining) - future))
            points = sorted(front)
            early = options.stage_size_tolerance
            # В конце допускаем более широкий диапазон; резервируем минимум 1 клетку/этап.
            tol = max(early, run.greedy_tail_tolerance) if future == 1 else early
            lo, hi = seq.stage_bounds((nominal,), tol)[0]
            hi = min(hi, len(remaining) - future)
            tried = set()
            last = -math.inf
            for trial in range(1, run.greedy_trials + 1):
                if time.monotonic() - last >= cfg.progress_interval_s:
                    logger.info("Жадный {}/{} | кандидат {}/{} | CWEF={} | кеш={}", i + 1,
                                len(counts), trial, run.greedy_trials, scorer.calculated, scorer.hits)
                    last = time.monotonic()
                # Два блока здесь проверяют только геометрию остатка, не его CWEF.
                n = len(remaining)
                gene = random_gene(points, rng)
                ranges = ((lo, hi), (1, n - future)) if future == 1 and best is None else ((lo, hi),)
                for sub, (low, high) in enumerate(ranges):
                    if (low, high) != (lo, hi) and best is not None:
                        break
                    cut = seq.decode(remaining, front, (nominal, n - nominal), (gene,),
                                     ((low, high), (n - high, n - low)))
                    if cut is None:
                        store.event(i + 1, trial, sub, "decode", plan=plan, seed=gene[0],
                                    note=f"No connected cut in range {low}-{high} cells")
                        continue
                    if cut[0] in tried:
                        store.event(i + 1, trial, sub, "duplicate", plan=[*plan, cut[0]])
                        continue
                    part = cut[0]
                    tried.add(part)
                    value = scorer.score(built | part)
                    improved = value is not None and (best_score is None or value < best_score)
                    if improved:
                        best, best_score = part, value
                    status = "invalid" if value is None else "leader" if improved else "worse"
                    store.event(i + 1, trial, sub, status, plan=[*plan, part], score=value,
                                best=best_score, active=i + 1,
                                note="Compared with candidates for this stage")
        if best is None:
            if future:
                logger.warning("Связный промежуточный срез не найден. Заходка {} закрывает "
                               "весь остаток; фактическое число этапов меньше заданного.", i + 1)
            best, best_score = remaining, scorer.score(target)
            if best_score is None:
                raise RuntimeError("Штатная геометрия отвергла целевой контур.")
        built |= best
        plan.append(best)
        state.update(step=len(plan), plan=plan, best_score=best_score, rng=rng.getstate(),
                     done=(built == target))
        state["history"].append(dict(stage=len(plan), cell_count=len(best), score=best_score))
        store.commit_state(state, checkpoint=(i + 1, run.greedy_trials + 1, dict(
            plan=plan, score=best_score, note="Stage accepted",
        )))
        publish(state)
        logger.info("Жадный: принята заходка {} | клеток={} | score={:.8g}",
                    len(plan), len(best), best_score)


def run_search(cfg, options, run, objective=seq.mean_normal_cwef):
    if run.mode not in {"ga", "greedy"} or run.greedy_trials < 1:
        raise ValueError("Нужны mode=ga/greedy и положительное greedy_trials.")
    if (run.make_animation or run.render_only) and min(cfg.figure_width, cfg.figure_height) < 320:
        raise ValueError("Для GIF нужны figure_width и figure_height не меньше 320.")
    for value in (options.stage_size_tolerance, run.greedy_tail_tolerance):
        seq.stage_bounds((1,), value)
    if not 0 <= options.immigrant_probability <= 1:
        raise ValueError("immigrant_probability должно лежать в [0, 1].")
    cfg = seq.normalize_paths(cfg)
    path = (seq.PROJECT_ROOT / Path(options.target_path).expanduser()).resolve() if options.target_path else None
    if path:
        cfg = replace(cfg, target_area_m2=2 * cfg.cell_side_m**2)
    cfg.validate()
    wal = Path(f"{cfg.weather_path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f"Погодная БД содержит непогашенный WAL: {wal}")
    directory = (seq.PROJECT_ROOT / Path(run.directory).expanduser()).resolve()
    if run.render_only and not (directory / "checkpoint.sqlite3").is_file():
        raise FileNotFoundError(directory / "checkpoint.sqlite3")
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Этот каталог уже занят другим расчётом: {directory}") from exc
        seq.configure_logging(cfg, directory)
        return run_locked(cfg, options, run, objective, path, directory)


def run_locked(cfg, options, run, objective, path, directory):
    geometry = seq.SequenceGeometry(cfg, build_default_target=path is None)
    physics_cfg = seq.prepare_bathymetry(cfg)
    code_files = (
        "sequential_resume.py", "sequential_animation.py", "sequential.py", "geometry.py", "anchored.py",
        "optimize.py", "physics.py", "bathymetry.py", "bathymetry_policy.py", "config.py",
    )
    settings = asdict(physics_cfg)
    presentation = {
        key: settings.pop(key) for key in ("animation_fps", "figure_width", "figure_height")
    }
    manifest = json.loads(dump(dict(
        format=2, config=settings, options={**asdict(options), "target_path": path},
        mode=run.mode, greedy_trials=run.greedy_trials, greedy_tail_tolerance=run.greedy_tail_tolerance,
        objective=[run.objective_id, objective.__module__, getattr(objective, "__qualname__", type(objective).__name__)],
        inputs=[path_stamp(p) for p in (
            path, cfg.coastline_path, cfg.weather_path, cfg.land_path, physics_cfg.bathymetry_path)],
        implementation={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in code_files},
    )))

    def feature(obj, **props):
        return dict(type="Feature", geometry=mapping(geometry.wgs84(obj)), properties=props)

    def collection(features):
        return dict(type="FeatureCollection", features=features)

    with closing(sqlite3.connect(directory / "checkpoint.sqlite3", timeout=10)) as db:
        store = Store(db, manifest)
        saved = store.read("target")
        if saved is not None:
            target = cells(saved)
        elif path is not None:
            target = seq.read_target(path, geometry)
        else:
            target = seq.GeneticBuilder(cfg, geometry, random.Random(cfg.random_seed)).initial()
        geometry.target = target
        starts = frozenset(c for c in target if cfg.connection == "island" or geometry.units[c] == 1)
        if len(target) < 2 or not starts or not geometry.valid(target):
            raise ValueError("Целевой контур не соответствует сетке, связности или режиму connection.")
        geometry.scenario(target)  # Геометрия проверяется без расчёта волн.
        counts, weights = seq.stage_parameters(options, len(target))
        area = sum(geometry.units[c] for c in target) * cfg.cell_side_m**2 / 2
        cfg = replace(physics_cfg, target_area_m2=area)
        geometry.cfg = cfg
        if saved is None:
            with db:
                store.put("target", target)
        seq.write_json(directory / "config.json", {
            **manifest, "nominal_cell_counts": counts, "animation": presentation,
        })
        seq.write_json(directory / "target.geojson",
                       collection([feature(geometry._fill(target), kind="reclamation", area_m2=area)]))

        def publish(state):
            seq.write_json(directory / "evolution.json", state["history"])
            rows = [cells(row) for row in state["plan"]]
            if not rows:
                return
            features = []
            for i, (part, built) in enumerate(zip(rows, cumulative(rows)), 1):
                score = store.get_score(digest(built))[0]
                features.append(feature(
                    geometry._fill(part), kind="stage", stage=i, cell_count=len(part),
                    area_m2=sum(geometry.units[c] for c in part) * cfg.cell_side_m**2 / 2,
                    water_area_m2=math.fsum(geometry.water_areas[c] for c in part), score=score,
                ))
            seq.write_json(directory / "best_so_far.geojson", collection(features))
            if state["done"]:
                seq.write_json(directory / "stages.geojson", collection(features))
                final = frozenset().union(*rows)
                seq.write_json(directory / "best.geojson", collection([
                    feature(geometry._fill(final), kind="reclamation", area_m2=area),
                ]))
                seq.write_json(directory / "summary.json", dict(
                    mode=run.mode, stage_count=len(rows), requested_stage_count=len(counts),
                    target_area_m2=area, score=state["best_score"],
                    score_kind="weighted_plan" if run.mode == "ga" else "final_state",
                    weighted_plan_score=math.fsum(
                        w * f["properties"]["score"] for w, f in zip(weights, features)
                    ) if len(rows) == len(counts) else None,
                    stages=[f["properties"] for f in features],
                ))

        state = store.read("state")
        if state is None:
            state = dict(step=0, population=[], plan=[], best_score=None, history=[], done=False,
                         rng=random.Random(cfg.random_seed).getstate())
            store.commit_state(state)
        else:
            logger.info("Возобновление {} | завершено шагов: {}", directory, state["step"])
        publish(state)
        if not state["done"] and not run.render_only:
            evaluator = seq.Evaluator(cfg, geometry)
            try:
                scorer = Scorer(store, evaluator, geometry, objective)
                store.event(0, 0, 0, "baseline", score=store.get_score(digest(frozenset()))[0])
                if run.mode == "ga":
                    evolve(cfg, options, target, starts, counts, weights, scorer, store, state, publish)
                else:
                    greedy(cfg, options, run, target, starts, counts, scorer, store, state, publish)
            finally:
                evaluator.close()
        if run.make_animation or run.render_only:
            def frames():
                for event in store.events():
                    yield {**event, "mode": run.mode}
                if state["done"]:
                    plan = [cells(part) for part in state["plan"]]
                    for i, built in enumerate(cumulative(plan), 1):
                        yield dict(mode=run.mode, step=i, status="final",
                                   plan=plan[:i], active=i,
                                   score=store.get_score(digest(built))[0],
                                   note=f"Final construction: stage {i}/{len(plan)}")

            output = directory / "evolution.gif"
            logger.info("Сборка GIF из журнала | {}", output)
            count = animation(output, frames(), geometry, target, fps=cfg.animation_fps,
                              width=cfg.figure_width, height=cfg.figure_height)
            if count:
                logger.info("Анимация: {} | кадров={}", output, count)
        logger.complete()
    return directory


def main():
    root = seq.PROJECT_ROOT
    mode = "ga"  # Для отдельного жадного запуска: "greedy".
    mode = "greedy"  # Для отдельного жадного запуска: "greedy".
    render_only = False  # True: собрать GIF из журнала, не продолжая поиск.
    cfg = seq.Config(
        coastline_path=root / "Kaliningrad/data/coastline/kaliningrad_OFFSET_merged.geojson",
        weather_path=root / "Kaliningrad/data/db/weather_compact.db",
        land_path=None, bathymetry_path=None, bathymetry_auto_download=True,
        bathymetry_cache_directory=root / "dalain/data/bathymetry",
        center_lon=19.95, center_lat=54.74, target_area_m2=1_000_000, cell_side_m=20,
        search_radius_m=1_000, evaluation_radius_m=1_250, connection="attached", sea_side="right",
        start_date="2025-01-01", end_date="2025-12-31", direction_count=36,
        point_spacing_m=50, tangent_delta_m=20, epochs=2, generations=12,
        population_size=12, construction_attempts=300, random_seed=42, progress_interval_s=5,
        animation_fps=8, figure_width=850, figure_height=720,
    )
    options = seq.SequenceConfig(target_path=None, stage_count=5, stage_size_tolerance=0.10,
                                 stage_cell_counts=None, stage_weights=None)
    run = RunOptions(directory=root / "dalain/output/resumable_animated" / mode,
                     mode=mode, greedy_trials=48, greedy_tail_tolerance=0.30,
                     make_animation=True, render_only=render_only)
    run_search(cfg, options, run)


if __name__ == "__main__":
    main()
