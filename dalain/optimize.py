from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np

from loguru import logger

from shapely.geometry import (
    LineString,
    Point,
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


# ============================================================
# РЕЗУЛЬТАТ ОЦЕНКИ
# ============================================================


@dataclass
class Evaluation:
    scenario: object
    metrics: list

    @property
    def mean_cwef(self):
        if not self.metrics:
            raise ValueError(
                "Нельзя вычислить среднее CWEF: "
                "нет расчётных точек"
            )

        return float(
            np.mean([
                row["mean_CWEF_Wm"]
                for row in self.metrics
            ])
        )


def mean_normal_cwef(result, baseline):
    """Среднее по точкам от среднего дневного нормального CWEF."""
    return result.mean_cwef


# ============================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================


def shape_id(cells):
    encoded = json.dumps(
        sorted(cells),
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
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
        f"Объект {type(value).__name__} "
        "не поддерживает сериализацию JSON"
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
    """Снимок явных констант текущего anchored.py."""
    allowed_types = (
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
        name: value
        for name, value in vars(mutation_module).items()
        if name.isupper()
        and isinstance(value, allowed_types)
    }


def normalize_paths(cfg):
    root = Path(__file__).resolve().parents[1]

    def absolute(value):
        if value is None:
            return None

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = root / path

        return path.resolve()

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
    """Проверка инвариантов после параллельной генерации."""
    if len(candidate) != len(parent):
        raise InvalidScenarioError(
            "Изменилось количество квадратов: "
            f"{len(parent)} -> {len(candidate)}"
        )

    if geometry.anchor_cell not in candidate:
        raise InvalidScenarioError(
            "Кандидат потерял якорную ячейку"
        )

    changed_count = len(
        parent ^ candidate
    )

    limit = builder.change_limit(parent)

    if changed_count > limit:
        raise InvalidScenarioError(
            "Превышен предел итогового изменения: "
            f"{changed_count} > {limit}"
        )

    if not geometry.valid(candidate):
        raise InvalidScenarioError(
            "Кандидат не прошёл геометрическую проверку"
        )


# ============================================================
# ПОСЛЕДОВАТЕЛЬНАЯ ОЦЕНКА CWEF
# ============================================================


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
            "{} | квадратов={} | площадь={:.1f} м² | "
            "исходных точек={} | новых={} | всего={}",
            label,
            len(cells),
            scenario.credited_area_m2,
            scenario.original_point_count,
            scenario.reclamation_point_count,
            len(scenario.sites),
        )

        metrics = []

        progress = Progress(
            f"{label}: CWEF",
            len(scenario.sites),
            self.cfg.progress_interval_s,
            self.cfg.point_log_every,
        )

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

        if len(metrics) != len(scenario.sites):
            raise RuntimeError(
                "Число результатов не совпадает "
                "с числом береговых точек"
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


# ============================================================
# СОХРАНЕНИЕ РЕШЕНИЙ
# ============================================================


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

        self.history_writer = csv.DictWriter(
            self.history_handle,
            fieldnames=self.columns,
        )
        self.history_writer.writeheader()

        self.accepted_writer = csv.DictWriter(
            self.accepted_handle,
            fieldnames=[
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
            ],
        )
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
        scenario = result.scenario

        solution_id = (
            f"solution_{self.sequence:08d}"
        )

        if parent_cells is None:
            removed = frozenset()
            added = frozenset()
        else:
            removed = (
                parent_cells - scenario.cells
            )
            added = (
                scenario.cells - parent_cells
            )

        anchor_inside = bool(
            scenario.cells
            and scenario.fill.contains(
                self.geometry.anchor_point
            )
        )

        context = {
            "solution_id": solution_id,
            "parent_solution_id": parent_solution_id,
            "epoch": epoch,
            "generation": generation,
            "individual": individual,
        }

        features = []

        def add(geometry, properties):
            if geometry.is_empty:
                return

            features.append({
                "type": "Feature",
                "geometry": mapping(
                    self.geometry.wgs84(geometry)
                ),
                "properties": {
                    **context,
                    **properties,
                },
            })

        add(
            self.geometry.anchor_point,
            {
                "kind": "anchor",
                "inside_reclamation": anchor_inside,
            },
        )

        for cell in sorted(scenario.cells):
            add(
                self.geometry.cells[cell],
                {
                    "kind": "cell",
                    "i": cell[0],
                    "j": cell[1],
                    "is_anchor_cell": (
                        cell == self.geometry.anchor_cell
                    ),
                    "area_units": (
                        self.geometry.units[cell]
                    ),
                    "is_added": cell in added,
                },
            )

        add(
            scenario.fill,
            {"kind": "reclamation"},
        )

        add(
            scenario.coast,
            {"kind": "coastline"},
        )

        normal_length = min(
            self.cfg.cell_side_m,
            self.cfg.point_spacing_m * 0.4,
        )

        for point_id, (site, metrics) in enumerate(
            zip(
                scenario.sites,
                result.metrics,
            )
        ):
            add(
                Point(site.x, site.y),
                {
                    "kind": "wave_point",
                    "point_id": point_id,
                    "normal_azimuth_deg": (
                        site.normal_deg
                    ),
                    **metrics,
                },
            )

            add(
                LineString([
                    (site.x, site.y),
                    (
                        site.x + site.nx * normal_length,
                        site.y + site.ny * normal_length,
                    ),
                ]),
                {
                    "kind": "normal",
                    "point_id": point_id,
                },
            )

        mutation_info = mutation_info or {}

        metadata = {
            **context,
            "shape_id": shape_id(
                scenario.cells
            ),
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
            "coast_length_m": (
                scenario.coast.length
            ),

            "removed_cell_count": len(removed),
            "added_cell_count": len(added),
            "changed_cell_count": (
                len(removed) + len(added)
            ),

            "anchor_inside": anchor_inside,
            "is_accepted": False,
            "cached": bool(cached),

            "mutation_seed": mutation_info.get(
                "seed"
            ),
            "mutation_seconds": mutation_info.get(
                "seconds",
                0.0,
            ),
            "mutation_stats": mutation_info.get(
                "stats",
                {},
            ),
            "mutation_error": mutation_info.get(
                "error"
            ),
        }

        path = (
            self.directory
            / "solutions"
            / f"{solution_id}.geojson"
        )

        write_json(
            path,
            {
                "type": "FeatureCollection",
                "metadata": metadata,
                "features": features,
            },
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

        metadata["is_accepted"] = True
        metadata["accepted_step"] = (
            self.accepted_count
        )

        write_json(
            path,
            content,
        )

        # Атомарная публикация лучшего решения.
        write_json(
            self.directory / "best.geojson",
            content,
        )

        self.accepted_writer.writerow({
            "accepted_step": self.accepted_count,
            "solution_id": metadata["solution_id"],
            "parent_solution_id": (
                metadata["parent_solution_id"]
            ),
            "epoch": metadata["epoch"],
            "generation": metadata["generation"],
            "score": metadata["score"],
            "mean_CWEF_Wm": metadata["mean_CWEF_Wm"],
            "area_m2": metadata["area_m2"],
            "removed_cell_count": (
                metadata["removed_cell_count"]
            ),
            "added_cell_count": (
                metadata["added_cell_count"]
            ),
            "changed_cell_count": (
                metadata["changed_cell_count"]
            ),
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
        features = []

        for cell in sorted(cells):
            if cell not in self.geometry.cells:
                continue

            features.append({
                "type": "Feature",
                "geometry": mapping(
                    self.geometry.wgs84(
                        self.geometry.cells[cell]
                    )
                ),
                "properties": {
                    "kind": "rejected_cell",
                    "i": cell[0],
                    "j": cell[1],
                },
            })

        path = (
            self.directory
            / "rejected"
            / f"rejected_{self.rejected_count:08d}.geojson"
        )

        write_json(
            path,
            {
                "type": "FeatureCollection",
                "metadata": {
                    "epoch": epoch,
                    "generation": generation,
                    "individual": individual,
                    "reason": str(reason),
                },
                "features": features,
            },
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
            {
                "epoch": epoch,
                "generation": generation,
                "parent_solution_id": (
                    parent_solution_id
                ),
                "backend": mutation_pool.backend,
                "workers": mutation_pool.workers,
                "elapsed_seconds": elapsed_seconds,
                "tasks": [
                    {
                        # Нулевая особь — сохранённый родитель.
                        "individual": result.index + 1,
                        "seed": result.seed,
                        "seconds": result.seconds,
                        "changed_cell_count": len(
                            parent_cells ^ result.cells
                        ),
                        "stats": result.stats,
                        "error": result.error,
                    }
                    for result in mutation_pool.last_results
                ],
            },
        )

    def frame(
        self,
        result,
        epoch,
        generation,
        history,
        color_max,
    ):
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        scenario = result.scenario

        figure = plt.figure(
            figsize=(
                self.cfg.figure_width / 100,
                self.cfg.figure_height / 100,
            ),
            dpi=100,
        )

        axis = figure.add_axes(
            [0.09, 0.29, 0.80, 0.63]
        )
        axis.set_aspect("equal")
        axis.set_facecolor("#e4f0f6")

        radius = (
            self.cfg.evaluation_radius_m
            + self.cfg.cell_side_m
        )

        view = Point(0, 0).buffer(radius)

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

        for cell in sorted(scenario.cells):
            axis.fill(
                *self.geometry.cells[cell].exterior.xy,
                facecolor="#ed974c",
                edgecolor="white",
                linewidth=0.4,
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
            [
                site.x
                for site in scenario.sites
            ],
            [
                site.y
                for site in scenario.sites
            ],
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

        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)
        axis.set_xlabel("Local east, m")
        axis.set_ylabel("Local north, m")

        label = (
            "Initial anchored solution"
            if generation < 0
            else (
                f"Epoch {epoch + 1}/{self.cfg.epochs}, "
                f"generation {generation + 1}/"
                f"{self.cfg.generations}"
            )
        )

        axis.set_title(
            f"{label}\n"
            f"Accepted mean normal CWEF = "
            f"{result.mean_cwef:.6g} W/m; "
            f"N = {len(scenario.cells)}"
        )

        curve = figure.add_axes(
            [0.12, 0.08, 0.75, 0.12]
        )

        curve.plot(
            history,
            color="#bf571b",
        )
        curve.set_xlabel(
            "Initial solution and generation frames"
        )
        curve.set_ylabel("Accepted objective")
        curve.grid(alpha=0.25)

        figure.savefig(
            self.directory
            / "frames"
            / f"frame_{self.frame_count:06d}.png"
        )

        plt.close(figure)
        self.frame_count += 1

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
            for index, path in enumerate(
                paths[1:],
                2,
            ):
                with Image.open(path) as image:
                    yield image.convert("RGB")

                if (
                    index % 10 == 0
                    or index == len(paths)
                ):
                    logger.info(
                        "GIF | {}/{} кадров",
                        index,
                        len(paths),
                    )

        with Image.open(paths[0]) as first:
            first.convert("RGB").save(
                self.directory / "evolution.gif",
                save_all=True,
                append_images=remaining(),
                duration=round(
                    1000 / self.cfg.animation_fps
                ),
                loop=0,
            )


# ============================================================
# ОСНОВНОЙ ЗАПУСК
# ============================================================


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

    mutation_total_seconds = 0.0
    evaluation_total_seconds = 0.0
    frame_total_seconds = 0.0

    try:
        logger.info(
            "DALAIN | параллельный поиск мутаций"
        )

        logger.info(
            "Исходная точка: lon={}, lat={}",
            cfg.center_lon,
            cfg.center_lat,
        )

        logger.info(
            "Критерий: среднее по береговым точкам "
            "от среднего дневного нормального CWEF"
        )

        with stage(
            "Проверка параметров и входных файлов"
        ):
            cfg.validate()

            wal = Path(
                str(cfg.weather_path) + "-wal"
            )

            if (
                wal.exists()
                and wal.stat().st_size
            ):
                raise RuntimeError(
                    "Погодная БД содержит "
                    f"непогашенный WAL: {wal}"
                )

            import PIL

        with stage(
            "Геометрия и проверка якоря"
        ):
            geometry = AnchoredGeometry(cfg)

        if not hasattr(
            geometry,
            "fixed_cell_count",
        ):
            raise RuntimeError(
                "Требуется актуальный anchored.py "
                "с поштучными переносами и "
                "фиксированным количеством квадратов"
            )

        from .bathymetry import prepare_bathymetry

        with stage(
            "Подготовка батиметрии"
        ):
            cfg = prepare_bathymetry(cfg)

        # Синхронизируем Config до создания snapshot workers.
        geometry.cfg = cfg

        objective_name = getattr(
            objective,
            "__name__",
            type(objective).__name__,
        )

        write_json(
            directory / "config.json",
            {
                **asdict(cfg),
                "objective": objective_name,
                "evolution": {
                    "mode": "anchored_1_plus_lambda",
                    "anchor_cell": [0, 0],
                    "random_epoch_restart": False,
                    "whole_shape_translation": False,
                    "operator_settings": (
                        operator_settings()
                    ),
                },
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
            objective_name,
        )

        with stage(
            "Исходный берег без намыва"
        ):
            evaluator.baseline = evaluator.calculate(
                frozenset(),
                "Исходный берег",
            )

        baseline = evaluator.baseline

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
            parent_cells = builder.initial()

        fixed_cell_count = len(
            parent_cells
        )

        if (
            geometry.fixed_cell_count
            != fixed_cell_count
        ):
            raise RuntimeError(
                "Первый намыв не зафиксировал "
                "количество квадратов"
            )

        with stage(
            "Оценка первого якорного намыва"
        ):
            parent_result = evaluator.evaluate(
                parent_cells,
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
            for row in baseline.metrics
        )

        recorder.frame(
            parent_result,
            0,
            -1,
            history,
            color_max,
        )

        # Не создаём больше workers, чем независимых
        # вариантов в одном поколении.
        actual_workers = min(
            MUTATION_WORKERS,
            max(
                1,
                cfg.population_size - 1,
            ),
        )

        with stage(
            "Инициализация постоянного пула мутаций"
        ):
            mutation_pool = ParallelMutationPool(
                builder,
                backend=MUTATION_BACKEND,
                workers=actual_workers,
            )

        write_json(
            directory / "parallelism.json",
            {
                "mutation_backend": (
                    mutation_pool.backend
                ),
                "mutation_workers": (
                    mutation_pool.workers
                ),
                "parallel_cwef": False,
                "population_order_preserved": True,
                "worker_writes_geojson": False,
            },
        )

        donor_pool = [
            parent_cells
        ]
        donor_scores = [
            parent_score
        ]

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
                key=lambda i: donor_scores[i],
            )

            return donor_pool[index]

        for epoch in range(cfg.epochs):
            logger.info(
                "ЭПОХА {}/{} | продолжение от {} | "
                "score={:.6f}",
                epoch + 1,
                cfg.epochs,
                parent_id,
                parent_score,
            )

            for generation in range(
                cfg.generations
            ):
                generation_started = (
                    time.perf_counter()
                )

                # Один фиксированный родитель
                # для всех вариантов поколения.
                source_cells = parent_cells
                source_result = parent_result
                source_score = parent_score
                source_id = parent_id

                mutation_started = (
                    time.perf_counter()
                )

                with stage(
                    "Параллельные варианты принятого решения"
                ):
                    # Всё случайное состояние задач
                    # задаётся в главном процессе.
                    jobs = [
                        (
                            choose_donor(),
                            rng.getrandbits(64),
                        )
                        for _ in range(
                            cfg.population_size - 1
                        )
                    ]

                    donors = [
                        donor
                        for donor, seed in jobs
                    ]

                    seeds = [
                        seed
                        for donor, seed in jobs
                    ]

                    variants = mutation_pool.generate(
                        parent=source_cells,
                        donors=donors,
                        seeds=seeds,
                        label=(
                            f"Э{epoch + 1}/"
                            f"П{generation + 1}: мутации"
                        ),
                    )

                    population = [
                        source_cells,
                        *variants,
                    ]

                mutation_seconds = (
                    time.perf_counter()
                    - mutation_started
                )

                mutation_total_seconds += (
                    mutation_seconds
                )

                recorder.save_mutation_batch(
                    epoch=epoch,
                    generation=generation,
                    parent_solution_id=source_id,
                    parent_cells=source_cells,
                    mutation_pool=mutation_pool,
                    elapsed_seconds=mutation_seconds,
                )

                if (
                    len(population)
                    != cfg.population_size
                ):
                    raise RuntimeError(
                        "Пул вернул неверный размер популяции"
                    )

                # Порядок worker-результатов соответствует
                # порядку variants, а не времени завершения.
                mutation_infos = [
                    None,
                    *[
                        {
                            "seed": result.seed,
                            "seconds": result.seconds,
                            "stats": result.stats,
                            "error": result.error,
                        }
                        for result in mutation_pool.last_results
                    ],
                ]

                winner_cells = source_cells
                winner_result = source_result
                winner_score = source_score
                winner_path = None
                winner_id = source_id

                scores = []

                evaluation_started = (
                    time.perf_counter()
                )

                for individual, cells in enumerate(
                    population
                ):
                    label = (
                        f"Э{epoch + 1}/"
                        f"П{generation + 1}/"
                        f"О{individual + 1}"
                    )

                    mutation_info = (
                        mutation_infos[individual]
                    )

                    cached = (
                        evaluator.cache.contains(cells)
                    )

                    try:
                        validate_population_candidate(
                            geometry,
                            builder,
                            source_cells,
                            cells,
                        )

                        if cells == source_cells:
                            # Родитель уже рассчитан.
                            result = source_result
                            cached = True
                        else:
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

                        cells = source_cells
                        population[individual] = cells
                        result = source_result
                        cached = True

                        mutation_info = dict(
                            mutation_info or {}
                        )
                        mutation_info["error"] = str(error)

                    score = checked_score(
                        objective,
                        result,
                        baseline,
                    )

                    path, solution_id = recorder.save(
                        result=result,
                        score=score,
                        epoch=epoch,
                        generation=generation,
                        individual=individual,
                        parent_cells=source_cells,
                        parent_solution_id=source_id,
                        cached=cached,
                        mutation_info=mutation_info,
                    )

                    scores.append(score)

                    logger.info(
                        "{} | CWEF={:.6f} Вт/м | "
                        "удалено={} | добавлено={} | "
                        "N={} | кеш={}",
                        label,
                        result.mean_cwef,
                        len(source_cells - cells),
                        len(cells - source_cells),
                        len(cells),
                        cached,
                    )

                    # При равенстве критерия
                    # сохраняется прежний родитель.
                    if score < winner_score:
                        winner_cells = cells
                        winner_result = result
                        winner_score = score
                        winner_path = path
                        winner_id = solution_id

                evaluation_seconds = (
                    time.perf_counter()
                    - evaluation_started
                )

                evaluation_total_seconds += (
                    evaluation_seconds
                )

                if winner_path is not None:
                    recorder.accept(
                        winner_path
                    )

                    logger.success(
                        "ПРИНЯТО | {} -> {} | "
                        "score {:.6f} -> {:.6f} | "
                        "удалено={} | добавлено={}",
                        source_id,
                        winner_id,
                        source_score,
                        winner_score,
                        len(source_cells - winner_cells),
                        len(winner_cells - source_cells),
                    )

                else:
                    logger.info(
                        "УЛУЧШЕНИЯ НЕТ | "
                        "сохранён родитель {}",
                        source_id,
                    )

                parent_cells = winner_cells
                parent_result = winner_result
                parent_score = winner_score
                parent_id = winner_id

                donor_pool = population
                donor_scores = scores

                history.append(
                    parent_score
                )

                frame_started = (
                    time.perf_counter()
                )

                with stage(
                    "Кадр принятой траектории"
                ):
                    recorder.frame(
                        parent_result,
                        epoch,
                        generation,
                        history,
                        color_max,
                    )

                frame_seconds = (
                    time.perf_counter()
                    - frame_started
                )

                frame_total_seconds += (
                    frame_seconds
                )

                unique_new = len(
                    set(population)
                    - {source_cells}
                )

                logger.info(
                    "ИТОГ ПОКОЛЕНИЯ | "
                    "родитель={} | score={:.6f} | "
                    "новых уникальных вариантов={} | "
                    "мутации={:.2f} с | "
                    "оценка и экспорт={:.2f} с | "
                    "кадр={:.2f} с | всего={:.2f} с",
                    parent_id,
                    parent_score,
                    unique_new,
                    mutation_seconds,
                    evaluation_seconds,
                    frame_seconds,
                    time.perf_counter() - generation_started,
                )

                evaluator.log_cache()

        with stage(
            "Анимация принятой траектории"
        ):
            recorder.animate()

        summary = {
            "objective": objective_name,

            "baseline_mean_CWEF_Wm": (
                baseline.mean_cwef
            ),
            "best_mean_CWEF_Wm": (
                parent_result.mean_cwef
            ),
            "best_score": parent_score,
            "best_solution_id": parent_id,

            "anchor_lon": cfg.center_lon,
            "anchor_lat": cfg.center_lat,
            "anchor_inside": (
                parent_result.scenario.fill.contains(
                    geometry.anchor_point
                )
            ),

            "initial_target_area_m2": (
                cfg.target_area_m2
            ),
            "best_credited_area_m2": (
                parent_result.scenario.credited_area_m2
            ),
            "best_area_difference_m2": (
                parent_result.scenario.credited_area_m2
                - cfg.target_area_m2
            ),
            "fixed_cell_count": fixed_cell_count,

            "solutions": recorder.sequence,
            "accepted_solutions": (
                recorder.accepted_count
            ),
            "rejected_scenarios": (
                recorder.rejected_count
            ),
            "frames": recorder.frame_count,

            "mutation_backend": (
                mutation_pool.backend
            ),
            "mutation_workers": (
                mutation_pool.workers
            ),

            "mutation_total_seconds": (
                mutation_total_seconds
            ),
            "evaluation_and_export_total_seconds": (
                evaluation_total_seconds
            ),
            "frame_total_seconds": (
                frame_total_seconds
            ),
            "elapsed_seconds": (
                time.perf_counter() - started
            ),

            "relative_change_percent": (
                100
                * (
                    parent_result.mean_cwef
                    / baseline.mean_cwef
                    - 1
                )
                if baseline.mean_cwef > 0
                else None
            ),
        }

        write_json(
            directory / "summary.json",
            summary,
        )

        logger.success(
            "ЗАВЕРШЕНО | "
            "решение={} | "
            "CWEF={:.6f} Вт/м | "
            "N={} | площадь={:.1f} м²",
            parent_id,
            parent_result.mean_cwef,
            len(parent_cells),
            parent_result.scenario.credited_area_m2,
        )

        logger.success(
            "Результаты: {}",
            directory,
        )

        return directory

    except BaseException:
        logger.exception(
            "Расчёт прерван. "
            "Промежуточные результаты: {}",
            directory,
        )
        raise

    finally:
        try:
            if mutation_pool is not None:
                logger.info(
                    "Закрытие пула мутаций..."
                )
                mutation_pool.close()

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