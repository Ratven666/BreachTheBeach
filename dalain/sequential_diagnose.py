"""Диагностика неудачного первого поколения без погоды, волн и записи в SQLite."""

from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import random
import sqlite3
import sys

from loguru import logger

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dalain import sequential as seq
from dalain import sequential_resume as resume


# Настройки для запуска кнопкой Run в IDE.
# Укажите папку с checkpoint.sqlite3, если нужен конкретный запуск.
# None: найти последнюю подходящую папку в AUTO_SEARCH_ROOT.
DEFAULT_RUN_DIRECTORY = None
AUTO_SEARCH_ROOT = seq.PROJECT_ROOT / "dalain" / "output" / "resumable_fast"
DEFAULT_SEARCH = True


def latest_run(root):
    candidates = []
    for checkpoint in root.rglob("checkpoint.sqlite3"):
        try:
            files = (checkpoint, checkpoint.with_name(checkpoint.name + "-wal"))
            stamp = max(p.stat().st_mtime_ns for p in files if p.exists())
            candidates.append((stamp, checkpoint))
        except (OSError, ValueError):
            continue
    for _, checkpoint in sorted(candidates, reverse=True):
        try:
            with closing(sqlite3.connect(f"{checkpoint.resolve().as_uri()}?mode=ro",
                                         uri=True)) as db:
                meta = {k: resume.unpack(v) for k, v in db.execute(
                    "SELECT key, val FROM meta WHERE key IN ('manifest', 'target', 'state')"
                )}
                first_attempt = db.execute("SELECT 1 FROM events WHERE step=1 LIMIT 1").fetchone()
            if (meta.get("manifest", {}).get("mode") == "ga"
                    and meta.get("state", {}).get("step") == 0
                    and meta.get("target") and first_attempt):
                return checkpoint.parent
        except (OSError, ValueError, sqlite3.Error):
            continue
    raise ValueError(
        f"В {root} не найден запуск ГА с незавершённым первым поколением. "
        "Укажите папку с checkpoint.sqlite3 в DEFAULT_RUN_DIRECTORY "
        "или передайте её аргументом."
    )


def read_run(directory):
    checkpoint = directory / "checkpoint.sqlite3"
    with closing(sqlite3.connect(f"{checkpoint.as_uri()}?mode=ro", uri=True)) as db:
        meta = {k: resume.unpack(v) for k, v in db.execute(
            "SELECT key, val FROM meta WHERE key IN ('manifest', 'target', 'state')"
        )}
        stats = Counter()
        for (blob,) in db.execute("SELECT payload FROM events WHERE step=1"):
            event = resume.unpack(blob)
            status = event["status"]
            if status == "invalid":
                status = f"invalid: этап {event.get('active', '?')}"
            stats[status] += 1
    if not {"manifest", "target", "state"} <= meta.keys():
        raise ValueError("В контрольной точке не записаны настройки, контур или состояние.")
    print("Журнал первого шага:", dict(stats) or "событий нет", flush=True)
    return meta


def inspect_batch(cfg, options, geometry, target, starts, candidates, cache):
    counts, _ = seq.stage_parameters(options, len(target))
    bounds = seq.stage_bounds(counts, options.stage_size_tolerance)
    points = sorted(target)
    rejected, seen = Counter(), set()
    for attempt, _, plan, key, _ in resume.ga_proposals(
        replace(cfg, construction_attempts=candidates), options, target, starts, counts,
        bounds, points, len(counts) - 1, [], random.Random(cfg.random_seed),
    ):
        if plan is None:
            rejected["декодер не нашёл связное разбиение"] += 1
            continue
        if key in seen:
            rejected["дубликат плана"] += 1
            continue
        seen.add(key)
        for stage, built in enumerate(resume.cumulative(plan), 1):
            state_key = resume.digest(built)
            if state_key not in cache:
                try:
                    geometry.scenario(built)
                except seq.InvalidScenarioError as exc:
                    cache[state_key] = str(exc)
                else:
                    cache[state_key] = None
            if cache[state_key] is not None:
                rejected[f"этап {stage}: {cache[state_key]}"] += 1
                break
        else:
            return (attempt, tuple(map(len, plan))), rejected
    return None, rejected


def diagnose(directory, search):
    meta = read_run(directory)
    manifest = meta["manifest"]
    if manifest["mode"] != "ga" or meta["state"]["step"] != 0:
        raise ValueError("Эта проверка предназначена для ГА, не сохранившего первое поколение.")
    cfg = seq.normalize_paths(seq.Config(**manifest["config"]))
    options = seq.SequenceConfig(**manifest["options"])
    target = resume.cells(meta["target"])
    default_target = options.target_path is None
    grid_cfg = cfg if default_target else replace(cfg, target_area_m2=2 * cfg.cell_side_m**2)
    geometry = seq.SequenceGeometry(grid_cfg, default_target)
    geometry.target = target
    missing = target.difference(geometry.cells)
    if missing:
        raise ValueError(f"На текущей сетке отсутствуют {len(missing)} клеток цели.")
    area = sum(geometry.units[cell] for cell in target) * cfg.cell_side_m**2 / 2
    geometry.cfg = replace(cfg, target_area_m2=area)
    starts = frozenset(c for c in target if cfg.connection == "island" or geometry.units[c] == 1)
    geometry.scenario(target)
    print(f"Цель: {len(target)} клеток, доступных стартов: {len(starts)}.", flush=True)
    counts, _ = seq.stage_parameters(options, len(target))
    print(f"Заходок: {len(counts)}; размеры в клетках: {tuple(counts)}; "
          f"допустимые диапазоны: {seq.stage_bounds(counts, options.stage_size_tolerance)}.",
          flush=True)
    cache = {resume.digest(target): None}
    variants = [(options, cfg.random_seed, cfg.construction_attempts)]
    if search:
        # Сохраняем число заходок. Сначала меняем только seed, затем при
        # необходимости пробуем допуск 30% или 45%. Проверки не ослабляем.
        variants.extend((replace(options, stage_size_tolerance=tolerance), seed,
                         max(300, cfg.construction_attempts))
                        for tolerance in dict.fromkeys((options.stage_size_tolerance, 0.30, 0.45))
                        if tolerance >= options.stage_size_tolerance
                        for seed in range(cfg.random_seed + 1, cfg.random_seed + 4))
    for opts, seed, limit in variants:
        print(f"\nseed={seed}, допуск={opts.stage_size_tolerance:.0%}, попыток до {limit}",
              flush=True)
        result, reasons = inspect_batch(replace(cfg, random_seed=seed), opts, geometry,
                                        target, starts, limit, cache)
        for reason, count in reasons.most_common(10):
            print(f"  {count:>4}  {reason}")
        if result is not None:
            attempt, sizes = result
            print(f"План прошёл проверку берега и нормалей: попытка {attempt}, размеры {sizes}.")
            print(f"В main(): random_seed={seed}, construction_attempts={limit}, "
                  f"stage_size_tolerance={opts.stage_size_tolerance}.")
            print("При изменении настроек задайте новый каталог. CWEF здесь не проверялся.")
            return 0
    print("В проверенных попытках допустимого плана нет. Проверки геометрии не отключались.")
    return 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", help="Папка с checkpoint.sqlite3")
    parser.add_argument("--search", action=BooleanOptionalAction, default=DEFAULT_SEARCH,
                        help="Проверить также другие seed и допуски")
    args = parser.parse_args()
    logger.remove()
    try:
        directory = args.directory or DEFAULT_RUN_DIRECTORY
        if directory is None:
            directory = latest_run(AUTO_SEARCH_ROOT)
        directory = (seq.PROJECT_ROOT / Path(directory).expanduser()).resolve()
        print(f"Папка расчёта: {directory}", flush=True)
        return diagnose(directory, args.search)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Диагностика остановлена: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
