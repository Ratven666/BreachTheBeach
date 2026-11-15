"""Последовательная оценка CWEF и эволюция с переменным числом потомков."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import csv
import hashlib
import json
import math
import random
import shutil
import time

import numpy as np

from loguru import logger

from shapely.geometry import (
    Point,
    LineString,
    mapping,
)

from . import anchored as mutation_module

from .anchored import (
    AnchoredGeometry,
    GeneticBuilder,
)

from .geometry import (
    InvalidScenarioError,
    geometry_parts,
)

from .logging_utils import (
    configure_logging,
    stage,
    Progress,
)

from .parallel_mutations import (
    ParallelMutationPool,
    MUTATION_BACKEND,
    MUTATION_WORKERS,
)


MAX_EMPTY_GENERATIONS = 3


@dataclass
class Evaluation:
    scenario: object
    metrics: list

    @property
    def mean_cwef(self):
        if not self.metrics:
            raise ValueError(
                "Нет расчётных точек"
            )

        return float(
            np.mean([
                row["mean_CWEF_Wm"]
                for row in self.metrics
            ])
        )


def mean_normal_cwef(result, baseline):
    return result.mean_cwef


def shape_id(cells):
    data = json.dumps(
        sorted(cells),
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(
        data
    ).hexdigest()[:20]


def json_default(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (set, frozenset)):
        return sorted(value)

    raise TypeError(
        "Не поддерживается JSON: "
        f"{type(value).__name__}"
    )


def write_json(path, content):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            content,
            default=json_default,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def operator_settings():
    types = (
        str,
        int,
        float,
        bool,
        tuple,
        list,
        dict,
        type(None),
    )

    return {
        key: value
        for key, value in vars(mutation_module).items()
        if key.isupper()
        and isinstance(value, types)
    }


def normalize_paths(cfg):
    root = Path(__file__).resolve().parents[1]

    def absolute(value):
        if value is None:
            return None

        path = Path(value).expanduser()

        return (
            path
            if path.is_absolute()
            else root / path
        ).resolve()

    names = (
        "coastline_path",
        "weather_path",
        "land_path",
        "bathymetry_path",
        "output_directory",
        "bathymetry_cache_directory",
    )

    return replace(
        cfg,
        **{
            name: absolute(
                getattr(cfg, name)
            )
            for name in names
        },
    )


def checked_score(objective, result, baseline):
    score = float(
        objective(result, baseline)
    )

    if not math.isfinite(score):
        raise ValueError(
            "Целевая функция вернула NaN или inf"
        )

    return score


def validate_population_candidate(
    geometry,
    builder,
    parent,
    candidate,
):
    if len(candidate) != len(parent):
        raise InvalidScenarioError(
            "Изменилось количество клеток"
        )

    if geometry.anchor_cell not in candidate:
        raise InvalidScenarioError(
            "Потеряна якорная клетка"
        )

    changed = len(
        parent ^ candidate
    )

    if changed and not (
        builder.minimum_change(parent)
        <= changed
        <= builder.change_limit(parent)
    ):
        raise InvalidScenarioError(
            "Изменение вне допустимого диапазона"
        )

    if not geometry.valid(candidate):
        raise InvalidScenarioError(
            "Недопустимая геометрия"
        )


class Evaluator:
    def __init__(self, cfg, geometry):
        from .physics import LRU, Physics

        self.cfg = cfg
        self.geometry = geometry

        self.physics = Physics(
            cfg,
            geometry,
        )

        self.cache = LRU(
            cfg.evaluation_cache_size
        )

        self.baseline = None

    def close(self):
        self.physics.close()

    def calculate(self, cells, label):
        with stage(
            f"{label}: берег и нормали"
        ):
            scenario = self.geometry.scenario(
                cells
            )

        logger.info(
            "{} | клеток={} | площадь={:.1f} м² | точек={}",
            label,
            len(cells),
            scenario.credited_area_m2,
            len(scenario.sites),
        )

        progress = Progress(
            f"{label}: CWEF",
            len(scenario.sites),
            self.cfg.progress_interval_s,
            self.cfg.point_log_every,
        )

        metrics = []

        for index, site in enumerate(
            scenario.sites,
            1,
        ):
            metrics.append(
                self.physics.point_metrics(
                    site,
                    scenario,
                )
            )

            progress.update(
                index,
                "последний mean CWEF="
                f"{metrics[-1]['mean_CWEF_Wm']:.6f} Вт/м",
            )

        result = Evaluation(
            scenario,
            metrics,
        )

        logger.info(
            "{} | среднее средних CWEF={:.6f} Вт/м",
            label,
            result.mean_cwef,
        )

        return result

    def evaluate(self, cells, label):
        return self.cache.get(
            cells,
            lambda: self.calculate(
                cells,
                label,
            ),
        )

    def log_cache(self):
        for name, cache in (
            ("решения", self.cache),
            ("погода", self.physics.weather_cache),
            ("лучи", self.physics.ray_cache),
            ("профили", self.physics.profile_cache),
        ):
            logger.info(
                "Кеш {} | hits={} | misses={} | entries={}",
                name,
                cache.hits,
                cache.misses,
                len(cache.items),
            )


class Recorder:
    def __init__(
        self,
        cfg,
        geometry,
        directory,
        objective,
    ):
        self.cfg = cfg
        self.geometry = geometry
        self.directory = directory
        self.objective = objective

        self.sequence = 0
        self.frame_count = 0
        self.accepted_count = 0
        self.rejected_count = 0

        for name in (
            "solutions",
            "frames",
            "rejected",
            "mutation_batches",
        ):
            (directory / name).mkdir()

        self.columns = [
            "solution_id",
            "parent_solution_id",
            "shape_id",
            "epoch",
            "generation",
            "individual",
            "score",
            "mean_CWEF_Wm",
            "area_m2",
            "area_difference_m2",
            "cell_count",
            "point_count",
            "removed_cell_count",
            "added_cell_count",
            "changed_cell_count",
            "anchor_inside",
            "cached",
            "mutation_seed",
            "mutation_seconds",
            "file",
        ]

        self.accepted_columns = [
            "accepted_step",
            "solution_id",
            "parent_solution_id",
            "epoch",
            "generation",
            "score",
            "mean_CWEF_Wm",
            "area_m2",
            "removed_cell_count",
            "added_cell_count",
            "changed_cell_count",
            "file",
        ]

        self.history_handle = (
            directory / "history.csv"
        ).open(
            "w",
            encoding="utf-8",
            newline="",
        )

        self.accepted_handle = (
            directory / "accepted.csv"
        ).open(
            "w",
            encoding="utf-8",
            newline="",
        )

        self.history_writer = csv.DictWriter(
            self.history_handle,
            fieldnames=self.columns,
        )

        self.accepted_writer = csv.DictWriter(
            self.accepted_handle,
            fieldnames=self.accepted_columns,
        )

        self.history_writer.writeheader()
        self.accepted_writer.writeheader()

    def close(self):
        self.history_handle.close()
        self.accepted_handle.close()

    def save(
        self,
        result,
        score,
        epoch,
        generation,
        individual,
        parent_cells=None,
        parent_solution_id=None,
        cached=False,
        mutation_info=None,
    ):
        geometry = self.geometry
        scenario = result.scenario

        solution_id = (
            f"solution_{self.sequence:08d}"
        )

        removed = (
            frozenset()
            if parent_cells is None
            else parent_cells - scenario.cells
        )

        added = (
            frozenset()
            if parent_cells is None
            else scenario.cells - parent_cells
        )

        context = dict(
            solution_id=solution_id,
            parent_solution_id=parent_solution_id,
            epoch=epoch,
            generation=generation,
            individual=individual,
        )

        features = []

        def add(shape, properties):
            if not shape.is_empty:
                features.append(
                    dict(
                        type="Feature",
                        geometry=mapping(
                            geometry.wgs84(shape)
                        ),
                        properties={
                            **context,
                            **properties,
                        },
                    )
                )

        anchor_inside = bool(
            scenario.cells
            and scenario.fill.contains(
                geometry.anchor_point
            )
        )

        add(
            geometry.anchor_point,
            dict(
                kind="anchor",
                inside_reclamation=anchor_inside,
            ),
        )

        for cell in sorted(scenario.cells):
            add(
                geometry.cells[cell],
                dict(
                    kind="cell",
                    i=cell[0],
                    j=cell[1],
                    area_units=geometry.units[cell],
                    is_anchor_cell=(
                        cell == geometry.anchor_cell
                    ),
                    is_added=cell in added,
                ),
            )

        for cell in sorted(removed):
            add(
                geometry.cells[cell],
                dict(
                    kind="removed_cell",
                    i=cell[0],
                    j=cell[1],
                ),
            )

        add(
            scenario.fill,
            dict(kind="reclamation"),
        )

        add(
            scenario.coast,
            dict(kind="coastline"),
        )

        length = min(
            self.cfg.cell_side_m,
            self.cfg.point_spacing_m * 0.4,
        )

        if (
            len(scenario.sites)
            != len(result.metrics)
        ):
            raise RuntimeError(
                "Число точек не совпадает "
                "с числом результатов CWEF"
            )

        for index, (site, metrics) in enumerate(
            zip(
                scenario.sites,
                result.metrics,
            )
        ):
            add(
                Point(site.x, site.y),
                dict(
                    kind="wave_point",
                    point_id=index,
                    normal_azimuth_deg=site.normal_deg,
                    **metrics,
                ),
            )

            add(
                LineString([
                    (site.x, site.y),
                    (
                        site.x + site.nx * length,
                        site.y + site.ny * length,
                    ),
                ]),
                dict(
                    kind="normal",
                    point_id=index,
                ),
            )

        info = mutation_info or {}

        metadata = {
            **context,

            "shape_id": shape_id(scenario.cells),
            "objective": self.objective,
            "score": float(score),
            "mean_CWEF_Wm": result.mean_cwef,

            "area_m2": scenario.credited_area_m2,
            "area_difference_m2": (
                scenario.credited_area_m2
                - self.cfg.target_area_m2
            ),
            "actual_water_area_m2": (
                scenario.actual_water_area_m2
            ),

            "cell_count": len(scenario.cells),
            "point_count": len(scenario.sites),
            "original_point_count": (
                scenario.original_point_count
            ),
            "reclamation_point_count": (
                scenario.reclamation_point_count
            ),
            "coast_length_m": scenario.coast.length,

            "removed_cell_count": len(removed),
            "added_cell_count": len(added),
            "changed_cell_count": (
                len(removed) + len(added)
            ),

            "anchor_inside": anchor_inside,
            "is_accepted": False,
            "cached": bool(cached),

            "mutation_seed": info.get("seed"),
            "mutation_seconds": info.get(
                "seconds",
                0.0,
            ),
            "mutation_stats": info.get(
                "stats",
                {},
            ),
            "mutation_error": info.get("error"),
        }

        path = (
            self.directory
            / "solutions"
            / f"{solution_id}.geojson"
        )

        write_json(
            path,
            dict(
                type="FeatureCollection",
                metadata=metadata,
                features=features,
            ),
        )

        self.history_writer.writerow({
            **{
                key: metadata[key]
                for key in self.columns
                if key != "file"
            },
            "file": path.name,
        })

        self.history_handle.flush()
        self.sequence += 1

        return path, solution_id

    def accept(self, path):
        content = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        metadata = content["metadata"]

        metadata.update(
            is_accepted=True,
            accepted_step=self.accepted_count,
        )

        write_json(
            path,
            content,
        )

        write_json(
            self.directory / "best.geojson",
            content,
        )

        row = {
            key: metadata[key]
            for key in self.accepted_columns
            if key != "file"
        }

        self.accepted_writer.writerow({
            **row,
            "file": path.name,
        })

        self.accepted_handle.flush()
        self.accepted_count += 1

    def reject(
        self,
        cells,
        reason,
        epoch,
        generation,
        individual,
    ):
        features = [
            dict(
                type="Feature",
                geometry=mapping(
                    self.geometry.wgs84(
                        self.geometry.cells[cell]
                    )
                ),
                properties=dict(
                    kind="rejected_cell",
                    i=cell[0],
                    j=cell[1],
                ),
            )
            for cell in sorted(cells)
            if cell in self.geometry.cells
        ]

        path = (
            self.directory
            / "rejected"
            / f"rejected_{self.rejected_count:08d}.geojson"
        )

        write_json(
            path,
            dict(
                type="FeatureCollection",
                features=features,
                metadata=dict(
                    epoch=epoch,
                    generation=generation,
                    individual=individual,
                    reason=str(reason),
                ),
            ),
        )

        self.rejected_count += 1

        logger.warning(
            "Кандидат отклонён | {} | {}",
            reason,
            path.name,
        )

    def save_mutation_batch(
        self,
        epoch,
        generation,
        parent_solution_id,
        parent_cells,
        mutation_pool,
        elapsed_seconds,
    ):
        def describe(row):
            return dict(
                index=row.index,
                seed=row.seed,
                seconds=row.seconds,
                changed_cell_count=len(
                    parent_cells ^ row.cells
                ),
                stats=row.stats,
                error=row.error,
            )

        path = (
            self.directory
            / "mutation_batches"
            / (
                f"epoch_{epoch:03d}_"
                f"generation_{generation:04d}.json"
            )
        )

        write_json(
            path,
            dict(
                epoch=epoch,
                generation=generation,
                parent_solution_id=parent_solution_id,
                backend=mutation_pool.backend,
                workers=mutation_pool.workers,
                elapsed_seconds=elapsed_seconds,

                sampling=mutation_pool.last_stats,

                tasks=[
                    describe(row)
                    for row in mutation_pool.last_results
                ],

                attempts=[
                    describe(row)
                    for row in mutation_pool.last_attempts
                ],
            ),
        )

    def frame(
        self,
        result,
        epoch,
        generation,
        history,
        color_max,
        parent_cells=None,
    ):
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        scenario = result.scenario

        figure, (axis, curve) = plt.subplots(
            2,
            1,
            figsize=(
                self.cfg.figure_width / 100,
                self.cfg.figure_height / 100,
            ),
            gridspec_kw={
                "height_ratios": [5, 1]
            },
            dpi=100,
        )

        try:
            axis.set_aspect("equal")
            axis.set_facecolor("#e4f0f6")

            radius = (
                self.cfg.evaluation_radius_m
                + self.cfg.cell_side_m
            )

            view = Point(0, 0).buffer(
                radius
            )

            for line in geometry_parts(
                self.geometry.original_coast.intersection(
                    view
                ),
                {"LineString"},
            ):
                axis.plot(
                    *line.xy,
                    color="#999999",
                    linewidth=0.7,
                )

            added = (
                frozenset()
                if parent_cells is None
                else scenario.cells - parent_cells
            )

            removed = (
                frozenset()
                if parent_cells is None
                else parent_cells - scenario.cells
            )

            for cell in sorted(scenario.cells):
                axis.fill(
                    *self.geometry.cells[cell].exterior.xy,
                    facecolor=(
                        "#4cab65"
                        if cell in added
                        else "#ed974c"
                    ),
                    edgecolor="white",
                    linewidth=0.3,
                )

            for cell in sorted(removed):
                axis.plot(
                    *self.geometry.cells[cell].exterior.xy,
                    color="#c43c39",
                    linewidth=0.6,
                )

            for line in geometry_parts(
                scenario.coast,
                {"LineString"},
            ):
                axis.plot(
                    *line.xy,
                    color="#263a47",
                    linewidth=0.8,
                )

            scatter = axis.scatter(
                [site.x for site in scenario.sites],
                [site.y for site in scenario.sites],
                c=[
                    row["mean_CWEF_Wm"]
                    for row in result.metrics
                ],
                s=12,
                cmap="viridis",
                vmin=0,
                vmax=max(1, color_max),
            )

            figure.colorbar(
                scatter,
                ax=axis,
                label="Mean normal CWEF, W/m",
                fraction=0.035,
            )

            axis.plot(
                *self.geometry.roi.exterior.xy,
                "--",
                color="#7b8b95",
                linewidth=0.7,
            )

            axis.plot(
                0,
                0,
                marker="*",
                color="red",
                markersize=10,
            )

            axis.set(
                xlim=(-radius, radius),
                ylim=(-radius, radius),
                xlabel="Local east, m",
                ylabel="Local north, m",
                title=(
                    f"Accepted | epoch={epoch + 1}, "
                    f"generation={generation + 1} | "
                    f"CWEF={result.mean_cwef:.6f} W/m"
                ),
            )

            curve.plot(
                history,
                color="#bf571b",
            )

            curve.set(
                xlabel="Generation",
                ylabel="Accepted objective",
            )

            curve.grid(alpha=0.25)

            figure.tight_layout()

            figure.savefig(
                self.directory
                / "frames"
                / f"frame_{self.frame_count:06d}.png"
            )

            self.frame_count += 1

        finally:
            plt.close(figure)

    def animate(self):
        from PIL import Image

        paths = sorted(
            (self.directory / "frames").glob(
                "frame_*.png"
            )
        )

        if not paths:
            return

        def remaining():
            for path in paths[1:]:
                with Image.open(path) as image:
                    yield image.convert("RGB")

        with Image.open(paths[0]) as image:
            image.convert("RGB").save(
                self.directory / "evolution.gif",
                save_all=True,
                append_images=remaining(),
                duration=round(
                    1000 / self.cfg.animation_fps
                ),
                loop=0,
            )


def run(cfg, objective=mean_normal_cwef):
    cfg = normalize_paths(cfg)

    directory = (
        cfg.output_directory
        / datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    configure_logging(
        cfg,
        directory,
    )

    evaluator = None
    recorder = None
    mutation_pool = None

    started = time.perf_counter()

    mutation_seconds = 0.0
    evaluation_seconds = 0.0

    completed = 0
    empty_generations = 0

    stop_reason = "completed"
    failed = False

    try:
        logger.info(
            "DALAIN | переменное поколение "
            "и параллельные мутации"
        )

        with stage(
            "Проверка параметров и входных файлов"
        ):
            cfg.validate()

            wal = Path(
                str(cfg.weather_path) + "-wal"
            )

            if wal.exists() and wal.stat().st_size:
                raise RuntimeError(
                    "Погодная БД содержит "
                    f"непогашенный WAL: {wal}"
                )

        with stage(
            "Геометрия и проверка якоря"
        ):
            geometry = AnchoredGeometry(cfg)

        from .bathymetry import prepare_bathymetry

        with stage(
            "Подготовка батиметрии"
        ):
            cfg = prepare_bathymetry(cfg)

        geometry.cfg = cfg

        name = getattr(
            objective,
            "__name__",
            type(objective).__name__,
        )

        write_json(
            directory / "config.json",
            {
                **asdict(cfg),
                "objective": name,
                "evolution": dict(
                    mode="anchored_1_plus_variable_lambda",
                    operator_settings=operator_settings(),
                    mutation_backend=MUTATION_BACKEND,
                    mutation_workers=MUTATION_WORKERS,
                    max_empty_generations=(
                        MAX_EMPTY_GENERATIONS
                    ),
                ),
            },
        )

        if cfg.bathymetry_path is not None:
            manifest = (
                cfg.bathymetry_path.parent
                / "manifest.json"
            )

            if manifest.is_file():
                shutil.copyfile(
                    manifest,
                    directory / "bathymetry_manifest.json",
                )

        evaluator = Evaluator(
            cfg,
            geometry,
        )

        recorder = Recorder(
            cfg,
            geometry,
            directory,
            name,
        )

        with stage(
            "Исходный берег без намыва"
        ):
            baseline = evaluator.calculate(
                frozenset(),
                "Исходный берег",
            )

            evaluator.baseline = baseline

        recorder.save(
            baseline,
            checked_score(
                objective,
                baseline,
                baseline,
            ),
            -1,
            -1,
            -1,
        )

        rng = random.Random(
            cfg.random_seed
        )

        builder = GeneticBuilder(
            cfg,
            geometry,
            rng,
        )

        with stage(
            "Построение первого якорного намыва"
        ):
            parent = builder.initial()

        with stage(
            "Оценка первого якорного намыва"
        ):
            parent_result = evaluator.evaluate(
                parent,
                "Первый якорный намыв",
            )

        parent_score = checked_score(
            objective,
            parent_result,
            baseline,
        )

        parent_path, parent_id = recorder.save(
            parent_result,
            parent_score,
            0,
            -1,
            0,
        )

        recorder.accept(
            parent_path
        )

        history = [
            parent_score
        ]

        color_max = max(
            row["mean_CWEF_Wm"]
            for result in (
                baseline,
                parent_result,
            )
            for row in result.metrics
        )

        recorder.frame(
            parent_result,
            0,
            -1,
            history,
            color_max,
        )

        donor_pool = [parent]
        donor_scores = [parent_score]

        mutation_pool = ParallelMutationPool(
            builder,
            backend=MUTATION_BACKEND,
            workers=MUTATION_WORKERS,
        )

        stop = False

        for epoch in range(cfg.epochs):
            for generation in range(cfg.generations):
                source = parent
                source_id = parent_id
                source_score = parent_score
                source_result = parent_result

                def choose_donor():
                    indices = [
                        rng.randrange(
                            len(donor_pool)
                        )
                        for _ in range(
                            cfg.tournament_size
                        )
                    ]

                    index = min(
                        indices,
                        key=lambda candidate: (
                            donor_scores[candidate]
                        ),
                    )

                    return donor_pool[index]

                donors = [
                    choose_donor()
                    for _ in range(
                        cfg.population_size - 1
                    )
                ]

                seeds = [
                    rng.getrandbits(64)
                    for _ in donors
                ]

                step_started = time.perf_counter()

                with stage(
                    "Параллельные варианты принятого решения"
                ):
                    variants = mutation_pool.generate(
                        source,
                        donors,
                        seeds,
                        label=(
                            f"Э{epoch + 1}/"
                            f"П{generation + 1}: мутации"
                        ),
                    )

                elapsed = (
                    time.perf_counter()
                    - step_started
                )

                mutation_seconds += elapsed

                recorder.save_mutation_batch(
                    epoch,
                    generation,
                    source_id,
                    source,
                    mutation_pool,
                    elapsed,
                )

                if (
                    len(variants)
                    > cfg.population_size - 1
                    or len(variants)
                    != len(mutation_pool.last_results)
                ):
                    raise RuntimeError(
                        "Пул вернул несогласованные "
                        "размеры результатов"
                    )

                if (
                    source in variants
                    or len(set(variants))
                    != len(variants)
                ):
                    raise RuntimeError(
                        "Пул вернул копии родителя "
                        "или дубликаты"
                    )

                logger.info(
                    "ПОПУЛЯЦИЯ | родитель=1 | новых={} | "
                    "всего={}/{}",
                    len(variants),
                    1 + len(variants),
                    cfg.population_size,
                )

                empty_generations = (
                    empty_generations + 1
                    if not variants
                    else 0
                )

                next_donors = [source]
                next_scores = [source_score]

                winner = (
                    source,
                    source_result,
                    source_score,
                    parent_path,
                    source_id,
                )

                step_started = time.perf_counter()

                for individual, (cells, info) in enumerate(
                    zip(
                        variants,
                        mutation_pool.last_results,
                    ),
                    1,
                ):
                    label = (
                        f"Э{epoch + 1}/"
                        f"П{generation + 1}/"
                        f"О{individual + 1}"
                    )

                    cached = evaluator.cache.contains(
                        cells
                    )

                    try:
                        validate_population_candidate(
                            geometry,
                            builder,
                            source,
                            cells,
                        )

                        result = evaluator.evaluate(
                            cells,
                            label,
                        )

                    except InvalidScenarioError as error:
                        recorder.reject(
                            cells,
                            error,
                            epoch,
                            generation,
                            individual,
                        )
                        continue

                    score = checked_score(
                        objective,
                        result,
                        baseline,
                    )

                    path, solution_id = recorder.save(
                        result,
                        score,
                        epoch,
                        generation,
                        individual,
                        parent_cells=source,
                        parent_solution_id=source_id,
                        cached=cached,
                        mutation_info=dict(
                            seed=info.seed,
                            seconds=info.seconds,
                            stats=info.stats,
                            error=info.error,
                        ),
                    )

                    next_donors.append(cells)
                    next_scores.append(score)

                    logger.info(
                        "{} | score={:.6f} | "
                        "удалено={} | добавлено={} | кеш={}",
                        label,
                        score,
                        len(source - cells),
                        len(cells - source),
                        cached,
                    )

                    if score < winner[2]:
                        winner = (
                            cells,
                            result,
                            score,
                            path,
                            solution_id,
                        )

                evaluation_seconds += (
                    time.perf_counter()
                    - step_started
                )

                (
                    parent,
                    parent_result,
                    parent_score,
                    parent_path,
                    parent_id,
                ) = winner

                if parent != source:
                    recorder.accept(
                        parent_path
                    )

                    logger.success(
                        "ПРИНЯТО | {} -> {} | "
                        "score {:.6f} -> {:.6f}",
                        source_id,
                        parent_id,
                        source_score,
                        parent_score,
                    )

                else:
                    logger.info(
                        "Родитель сохранён: улучшения нет"
                    )

                donor_pool = next_donors
                donor_scores = next_scores

                history.append(
                    parent_score
                )

                completed += 1

                recorder.frame(
                    parent_result,
                    epoch,
                    generation,
                    history,
                    color_max,
                    parent_cells=source,
                )

                evaluator.log_cache()

                if (
                    empty_generations
                    >= MAX_EMPTY_GENERATIONS
                ):
                    stop_reason = "no_new_variants"

                    logger.warning(
                        "{} поколений без новых форм; "
                        "нормальное завершение поиска",
                        empty_generations,
                    )

                    stop = True
                    break

            if stop:
                break

        recorder.animate()

        summary = dict(
            objective=name,
            solution_id=parent_id,
            shape_id=shape_id(parent),
            score=parent_score,

            mean_CWEF_Wm=parent_result.mean_cwef,
            baseline_mean_CWEF_Wm=baseline.mean_cwef,

            cell_count=len(parent),
            area_m2=(
                parent_result.scenario.credited_area_m2
            ),

            completed_generations=completed,
            stop_reason=stop_reason,

            mutation_seconds=mutation_seconds,
            evaluation_seconds=evaluation_seconds,
            seconds=time.perf_counter() - started,

            accepted_count=recorder.accepted_count,
            rejected_count=recorder.rejected_count,

            solution_file=parent_path.name,
        )

        write_json(
            directory / "summary.json",
            summary,
        )

        logger.success(
            "ЗАВЕРШЕНО | {} | score={:.6f} | {}",
            parent_id,
            parent_score,
            directory,
        )

        return directory

    except BaseException:
        failed = True

        logger.exception(
            "Расчёт прерван. "
            "Промежуточные результаты: {}",
            directory,
        )

        raise

    finally:
        try:
            if mutation_pool is not None:
                mutation_pool.close(
                    abort=failed
                )

        finally:
            try:
                if recorder is not None:
                    recorder.close()

            finally:
                try:
                    if evaluator is not None:
                        evaluator.close()

                finally:
                    logger.complete()