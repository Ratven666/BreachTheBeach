from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
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

from .geometry import (
    InvalidScenarioError,
    geometry_parts,
    is_connected,
    neighbours,
)

from .logging_utils import (
    configure_logging,
    stage,
    Progress,
)


@dataclass
class Evaluation:
    scenario: object
    metrics: list

    @property
    def mean_cwef(self):
        return float(
            np.mean(
                [
                    row["mean_CWEF_Wm"]
                    for row in self.metrics
                ]
            )
        )


def mean_normal_cwef(result, baseline):
    return result.mean_cwef


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
            f"{label}: береговая линия и нормали"
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

        progress = Progress(
            f"{label}: CWEF",
            len(scenario.sites),
            self.cfg.progress_interval_s,
            self.cfg.point_log_every,
        )

        # Отдельный список для текущего решения.
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


class GeneticBuilder:
    def __init__(self, cfg, geometry, rng):
        self.cfg = cfg
        self.geometry = geometry
        self.rng = rng

        self.center_water = {}

    def _simple_add(self, cell, cells):
        """Консервативная локальная защита от замыкания воды.

        Полная геометрическая проверка всё равно
        выполняется после построения кандидата.
        """
        i, j = cell

        ring = {
            (i + dx, j + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if dx or dy
        }

        free = set()
        side = self.cfg.cell_side_m

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

    def trim(self, source, retained):
        cells = set(source)

        while len(cells) > retained:
            candidates = [
                cell
                for cell in sorted(cells)
                if any(
                    other not in cells
                    for other in neighbours(cell)
                )
            ]

            self.rng.shuffle(
                candidates
            )

            for cell in candidates:
                reduced = cells - {cell}

                if (
                    not reduced
                    or not is_connected(reduced)
                ):
                    continue

                if self.cfg.connection == "attached":
                    if not any(
                        self.geometry.units[other] == 1
                        for other in reduced
                    ):
                        continue

                cells = reduced
                break

            else:
                break

        return frozenset(cells)

    def build(
        self,
        preferred=frozenset(),
        core=frozenset(),
    ):
        geometry = self.geometry

        for attempt in range(
            self.cfg.construction_attempts
        ):
            if attempt == 0 and core:
                cells = set(core)

            else:
                inherited = [
                    cell
                    for cell in geometry.seeds
                    if cell in preferred
                ]

                pool = (
                    inherited
                    if inherited
                    and self.rng.random() < 0.8
                    else geometry.seeds
                )

                cells = {
                    self.rng.choice(pool)
                }

            units = sum(
                geometry.units[cell]
                for cell in cells
            )

            if units > self.cfg.target_units:
                continue

            center_i = (
                sum(cell[0] for cell in cells)
                / len(cells)
            )
            center_j = (
                sum(cell[1] for cell in cells)
                / len(cells)
            )

            frontier = {
                other
                for cell in cells
                for other in neighbours(cell)
                if other in geometry.cells
                and other not in cells
            }

            while units < self.cfg.target_units:
                remaining = (
                    self.cfg.target_units
                    - units
                )

                available = [
                    cell
                    for cell in sorted(frontier)
                    if (
                        geometry.units[cell]
                        <= remaining
                    )
                ]

                def priority(cell):
                    shared = sum(
                        other in cells
                        for other in neighbours(cell)
                    )

                    radial = (
                        (cell[0] - center_i)**2
                        + (cell[1] - center_j)**2
                    ) / max(
                        1,
                        self.cfg.target_units / 2,
                    )

                    inherited_bonus = (
                        0.8 if cell in preferred else 0.0
                    )

                    return (
                        radial
                        - 0.3 * shared
                        - inherited_bonus
                        + self.rng.random() * 0.7
                    )

                available.sort(
                    key=priority
                )

                cell = next(
                    (
                        candidate
                        for candidate in available
                        if self._simple_add(
                            candidate,
                            cells,
                        )
                    ),
                    None,
                )

                if cell is None:
                    break

                cells.add(cell)
                units += geometry.units[cell]

                frontier.discard(cell)

                frontier.update(
                    other
                    for other in neighbours(cell)
                    if other in geometry.cells
                    and other not in cells
                )

            candidate = frozenset(cells)

            if geometry.valid(candidate):
                logger.debug(
                    "Кандидат без озёр построен "
                    "за {} попыток | квадратов={}",
                    attempt + 1,
                    len(candidate),
                )

                return candidate

        raise RuntimeError(
            "Не удалось построить связный намыв "
            "точной площади без замкнутых водных полостей. "
            "Проверьте площадь, размер квадрата и радиус."
        )

    def child(self, first, second):
        result = first

        if (
            self.rng.random()
            < self.cfg.crossover_probability
        ):
            result = self.build(
                first | second,
                self.trim(
                    first,
                    max(1, len(first) // 2),
                ),
            )

        if (
            self.rng.random()
            < self.cfg.mutation_probability
        ):
            dx, dy = self.rng.choice(
                (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                )
            )

            shifted = frozenset(
                (i + dx, j + dy)
                for i, j in result
            )

            if (
                self.rng.random() < 0.35
                and self.geometry.valid(shifted)
            ):
                return shifted

            retained = max(
                1,
                len(result)
                - max(
                    1,
                    math.ceil(
                        len(result)
                        * self.cfg.mutation_fraction
                    ),
                ),
            )

            result = self.build(
                result,
                self.trim(
                    result,
                    retained,
                ),
            )

        return result


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
        self.rejected_count = 0

        for name in (
            "solutions",
            "frames",
            "rejected",
        ):
            (directory / name).mkdir()

        self.handle = (
            directory / "history.csv"
        ).open(
            "w",
            encoding="utf-8",
            newline="",
        )

        self.columns = [
            "sequence",
            "epoch",
            "generation",
            "individual",
            "score",
            "mean_CWEF_Wm",
            "area_m2",
            "actual_water_area_m2",
            "cell_count",
            "point_count",
            "original_point_count",
            "reclamation_point_count",
            "coast_length_m",
            "cached",
            "file",
        ]

        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=self.columns,
        )

        self.writer.writeheader()

    def close(self):
        self.handle.close()

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

        path.write_text(
            json.dumps(
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
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        self.rejected_count += 1

        logger.warning(
            "Кандидат отклонён: {} | {}",
            reason,
            path.name,
        )

    def save(
        self,
        result,
        score,
        epoch,
        generation,
        individual,
        cached=False,
    ):
        scenario = result.scenario
        features = []

        context = {
            "epoch": epoch,
            "generation": generation,
            "individual": individual,
        }

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

        for cell in sorted(scenario.cells):
            add(
                self.geometry.cells[cell],
                {
                    "kind": "cell",
                    "i": cell[0],
                    "j": cell[1],
                    "area_units": (
                        self.geometry.units[cell]
                    ),
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

        length = min(
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
                        site.x + site.nx * length,
                        site.y + site.ny * length,
                    ),
                ]),
                {
                    "kind": "normal",
                    "point_id": point_id,
                },
            )

        metadata = {
            "sequence": self.sequence,
            **context,
            "objective": self.objective,
            "score": float(score),
            "mean_CWEF_Wm": result.mean_cwef,
            "area_m2": scenario.credited_area_m2,
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
            "cached": bool(cached),
        }

        path = (
            self.directory
            / "solutions"
            / f"solution_{self.sequence:08d}.geojson"
        )

        temporary = path.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "metadata": metadata,
                    "features": features,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

        self.writer.writerow({
            **{
                key: metadata[key]
                for key in self.columns
                if key != "file"
            },
            "file": path.name,
        })

        self.handle.flush()
        self.sequence += 1

        return path

    def frame(
        self,
        population,
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

        # Показываются только ячейки текущей популяции.
        for cells in population:
            for cell in sorted(cells):
                axis.fill(
                    *self.geometry.cells[
                        cell
                    ].exterior.xy,
                    color="#597ea6",
                    alpha=0.04,
                )

        for cell in sorted(scenario.cells):
            axis.fill(
                *self.geometry.cells[
                    cell
                ].exterior.xy,
                facecolor="#ed974c",
                edgecolor="white",
                linewidth=0.5,
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

        # Только точки текущего результата.
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

        length = min(
            self.cfg.cell_side_m,
            self.cfg.point_spacing_m * 0.3,
        )

        axis.quiver(
            [
                site.x
                for site in scenario.sites
            ],
            [
                site.y
                for site in scenario.sites
            ],
            [
                site.nx * length
                for site in scenario.sites
            ],
            [
                site.ny * length
                for site in scenario.sites
            ],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            alpha=0.5,
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
            0, 0, "+",
            color="red",
        )

        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)

        axis.set_xlabel("Local east, m")
        axis.set_ylabel("Local north, m")

        axis.set_title(
            f"Epoch {epoch + 1}/{self.cfg.epochs}, "
            f"generation {generation + 1}/{self.cfg.generations}\n"
            f"Mean normal CWEF = {result.mean_cwef:.6g} W/m; "
            f"points = {len(scenario.sites)}"
        )

        curve = figure.add_axes(
            [0.12, 0.08, 0.75, 0.12]
        )

        curve.plot(
            history,
            color="#bf571b",
        )

        curve.set_xlabel(
            "Generation frames across all epochs"
        )
        curve.set_ylabel("Global best")
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
                        "GIF: {}/{} кадров",
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
            "Целевая функция вернула NaN/inf"
        )

    return score


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

    started = time.perf_counter()
    evaluator = None
    recorder = None

    try:
        logger.info(
            "DALAIN | связный намыв без новых озёр"
        )

        logger.info(
            "Каталог результатов: {}",
            directory,
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

        from .bathymetry import prepare_bathymetry
        from .geometry import Geometry

        with stage("Подготовка батиметрии"):
            cfg = prepare_bathymetry(cfg)

        name = getattr(
            objective,
            "__name__",
            type(objective).__name__,
        )

        (directory / "config.json").write_text(
            json.dumps(
                {
                    **asdict(cfg),
                    "objective": name,
                    "geometry_algorithm": (
                        "local_land_no_lakes_stable_points_v2"
                    ),
                },
                default=str,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
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

        with stage(
            "Локальная суша, исходные точки и сетка"
        ):
            geometry = Geometry(cfg)

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
            "Исходное состояние без намыва"
        ):
            evaluator.baseline = (
                evaluator.calculate(
                    frozenset(),
                    "Исходный берег",
                )
            )

        baseline = evaluator.baseline

        recorder.save(
            baseline,
            checked_score(
                objective,
                baseline,
                baseline,
            ),
            -1, -1, -1,
        )

        logger.success(
            "Исходный CWEF={:.6f} Вт/м | точек={}",
            baseline.mean_cwef,
            len(baseline.scenario.sites),
        )

        color_max = max(
            row["mean_CWEF_Wm"]
            for row in baseline.metrics
        )

        best_cells = None
        best_score = math.inf
        best_mean = None

        history = []

        total = (
            cfg.epochs
            * cfg.generations
            * cfg.population_size
        )
        completed = 0

        for epoch in range(cfg.epochs):
            rng = random.Random(
                cfg.random_seed + epoch
            )

            builder = GeneticBuilder(
                cfg,
                geometry,
                rng,
            )

            with stage(
                f"Эпоха {epoch + 1}: начальная популяция"
            ):
                population = [
                    builder.build()
                    for _ in range(
                        cfg.population_size
                    )
                ]

                if best_cells is not None:
                    population[0] = best_cells

            for generation in range(
                cfg.generations
            ):
                scores = []
                generation_started = (
                    time.perf_counter()
                )

                logger.info(
                    "ПОКОЛЕНИЕ | эпоха {}/{} | {}/{}",
                    epoch + 1,
                    cfg.epochs,
                    generation + 1,
                    cfg.generations,
                )

                for individual in range(
                    cfg.population_size
                ):
                    label = (
                        f"Э{epoch + 1}/"
                        f"П{generation + 1}/"
                        f"О{individual + 1}"
                    )

                    # Геометрическая неоднозначность
                    # не завершает весь расчёт.
                    for retry in range(50):
                        cells = population[individual]

                        cached = (
                            evaluator.cache.contains(cells)
                        )

                        logger.info(
                            "{} | оценка {}/{} | "
                            "кеш={} | попытка={}",
                            label,
                            completed + 1,
                            total,
                            cached,
                            retry + 1,
                        )

                        try:
                            result = evaluator.evaluate(
                                cells,
                                label,
                            )
                            break

                        except InvalidScenarioError as error:
                            recorder.reject(
                                cells,
                                error,
                                epoch,
                                generation,
                                individual,
                            )

                            population[individual] = (
                                builder.build()
                            )

                    else:
                        raise RuntimeError(
                            "Не удалось получить геометрически "
                            "однозначного кандидата за 50 попыток. "
                            "Проверьте исходный берег."
                        )

                    score = checked_score(
                        objective,
                        result,
                        baseline,
                    )

                    path = recorder.save(
                        result,
                        score,
                        epoch,
                        generation,
                        individual,
                        cached,
                    )

                    scores.append(score)
                    completed += 1

                    if score < best_score:
                        best_cells = cells
                        best_score = score
                        best_mean = result.mean_cwef

                        shutil.copyfile(
                            path,
                            directory / "best.geojson",
                        )

                        logger.success(
                            "НОВЫЙ ЛУЧШИЙ | "
                            "CWEF={:.6f} Вт/м | точек={}",
                            best_mean,
                            len(result.scenario.sites),
                        )

                order = sorted(
                    range(len(population)),
                    key=lambda index: (
                        scores[index],
                        tuple(
                            sorted(
                                population[index]
                            )
                        ),
                    ),
                )

                index = order[0]

                result = evaluator.evaluate(
                    population[index],
                    "Кадр текущего поколения",
                )

                history.append(
                    best_score
                )

                with stage(
                    "Сохранение кадра поколения"
                ):
                    recorder.frame(
                        population,
                        result,
                        epoch,
                        generation,
                        history,
                        color_max,
                    )

                logger.info(
                    "ИТОГ | min={:.6f} | mean={:.6f} | "
                    "global={:.6f} | {:.1f} с",
                    min(scores),
                    float(np.mean(scores)),
                    best_score,
                    time.perf_counter() - generation_started,
                )

                evaluator.log_cache()

                if (
                    generation + 1
                    == cfg.generations
                ):
                    continue

                def tournament():
                    indices = [
                        rng.randrange(
                            len(population)
                        )
                        for _ in range(
                            cfg.tournament_size
                        )
                    ]

                    return population[
                        min(
                            indices,
                            key=lambda index: scores[index],
                        )
                    ]

                with stage(
                    "Кроссовер и мутация без замкнутых полостей"
                ):
                    children = [
                        population[index]
                        for index in order[
                            :cfg.elite_count
                        ]
                    ]

                    while (
                        len(children)
                        < cfg.population_size
                    ):
                        children.append(
                            builder.child(
                                tournament(),
                                tournament(),
                            )
                        )

                    population = children

        with stage("Создание анимации"):
            recorder.animate()

        summary = {
            "objective": name,
            "baseline_mean_CWEF_Wm": (
                baseline.mean_cwef
            ),
            "best_mean_CWEF_Wm": best_mean,
            "best_score": best_score,
            "target_area_m2": cfg.target_area_m2,
            "solutions": recorder.sequence,
            "rejected_scenarios": (
                recorder.rejected_count
            ),
            "frames": recorder.frame_count,
            "elapsed_seconds": (
                time.perf_counter() - started
            ),
            "relative_change_percent": (
                100
                * (
                    best_mean
                    / baseline.mean_cwef
                    - 1
                )
                if baseline.mean_cwef > 0
                else None
            ),
        }

        (directory / "summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        logger.success(
            "РАСЧЁТ ЗАВЕРШЁН | "
            "исходный CWEF={:.6f} | "
            "лучший CWEF={:.6f}",
            baseline.mean_cwef,
            best_mean,
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
        if recorder is not None:
            recorder.close()

        if evaluator is not None:
            evaluator.close()

        logger.complete()