"""
big_scale/pypeline/4_1_download_bathymetry.py

Архив батиметрии с использованием базовых загрузчиков проекта.

Загрузка:
- GEBCOOpenTopographyLoader;
- GEBCOOpenDAPLoader;
- EMODnetBathymetryLoader.

В пайплайне локально исправлено чтение GeoTIFF:
координаты значений — центры пикселей, а не внешние границы.

Возможности:
- пространственный кеш независимо от point_id/source_id/seq;
- повторное использование данных при изменении точек и линий;
- проверка структуры и читаемости NetCDF;
- индекс проверенного кеша;
- provenance источника и настроек загрузчика;
- подробная диагностика отклонённых тайлов;
- повтор только временных ошибок;
- атомарная публикация manifest.json;
- отсутствие изменений SQLite.

Ограничения:
- каждой точке нужен один полностью покрывающий её bbox тайл;
- мозаика нескольких частичных тайлов не строится;
- сохраняются ограничения BathymetryCoveragePlanner,
  включая отсутствие поддержки круга через антимеридиан;
- проверка finite-значений не доказывает отсутствие
  отдельных пропусков внутри всего покрытия.

Все настройки задаются константами ниже.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from pyproj import CRS
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from tqdm.auto import tqdm


# ==================== КОРЕНЬ ПРОЕКТА ====================

def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent

    for candidate in (directory, *directory.parents):
        if (
            candidate / "src/coastline/storage/models.py"
        ).is_file():
            return candidate

    raise RuntimeError(
        "Не найден корень BreachTheBeach."
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.bathymetry.archive.BathymetryArchive import (  # noqa: E402
    BathymetryArchiveBuilder,
    BathymetryArchiveConfig,
    BathymetryCoveragePlanner,
    _estimate_cells,
    _expand_bbox,
    _union_bbox,
)
from src.bathymetry.domain.models import BathymetryGrid  # noqa: E402
from src.bathymetry.errors import (  # noqa: E402
    BathymetryAuthenticationError,
    BathymetryConfigurationError,
)
from src.bathymetry.exporters.NetCDFBathymetryExporter import (  # noqa: E402
    NetCDFBathymetryExporter,
)
from src.bathymetry.loaders.EMODnetBathymetryLoader import (  # noqa: E402
    EMODnetBathymetryLoader,
)
from src.bathymetry.loaders.GEBCODownloadLoader import (  # noqa: E402
    GEBCODownloadLoader,
)
from src.bathymetry.loaders.gebco_opendap_loader import (  # noqa: E402
    GEBCOOpenDAPLoader,
)
from src.bathymetry.loaders.gebco_opentopography_loader import (  # noqa: E402
    GEBCOOpenTopographyLoader,
)


# ==================== НАСТРОЙКИ ====================

DATABASE_PATH = Path(
    "big_scale/data/db/coastline.db"
)

OUTPUT_DIR = Path(
    "big_scale/data/bathymetry"
)

# Другие каталоги ранее скачанных NetCDF.
# Файлы подключаются без копирования:
# эти каталоги должны оставаться доступными.
ADDITIONAL_CACHE_DIRS: list[Path] = []

# None — последний источник точек с рассчитанными fetch.
POINT_SOURCE_ID: int | None = None

# gebco | emodnet | auto
PROVIDER = "gebco"

# opentopography | opendap
GEBCO_BACKEND = "opentopography"

OPENTOPOGRAPHY_DEMTYPE = "GEBCOIceTopo"
OPENTOPOGRAPHY_URL = (
    "https://portal.opentopography.org/API/globaldem"
)

GEBCO_OPENDAP_URL = (
    "https://dap.ceda.ac.uk/thredds/dodsC/"
    "neodc/gebco/data/gebco_2025/GEBCO_2025.nc"
)

GEBCO_OPENDAP_ENGINE: str | None = None

EMODNET_URL = "https://ows.emodnet-bathymetry.eu/wcs"
EMODNET_COVERAGE = "emodnet:mean"

SAFETY_FACTOR = 1.05
PADDING_M = 1000.0

MAX_CELLS_PER_TILE = 10_000_000

# Только оценка числа ячеек.
# Для PROVIDER=auto/emodnet рекомендуется 3.75.
PLANNING_RESOLUTION_ARCSEC = 15.0

MAX_FETCH_M: float | None = None

MAX_LAT_SPAN_DEG = 4.0
MAX_LON_SPAN_DEG = 12.0

WORKERS = 1
QUEUE_MULTIPLIER = 1

DOWNLOAD_ATTEMPTS = 5
RETRY_INITIAL_S = 60.0
RETRY_MAX_S = 300.0
RETRY_JITTER_S = 10.0

PROVIDER_TIMEOUT_S = 600

GEBCO_MAX_RESOLUTION_ARCSEC = 15.0
EMODNET_MAX_RESOLUTION_ARCSEC = 3.75

# Допуск только на численную погрешность шага.
RESOLUTION_TOLERANCE_ARCSEC = 0.0001

VERIFY_ALL_CACHE_DATA_EVERY_RUN = False
VERIFY_BLOCK_CELLS = 1_000_000

# Явное разрешение прежних источников без provenance.
#
# Для использования ранее скачанного NOAA GEBCO_2020:
#
# ACCEPTED_LEGACY_SOURCES = {
#     "GEBCO_2020_NOAA_ERDDAP": 15.0,
# }
#
# Это разрешает смешение разных источников/версий.
# Фактический источник сохраняется для каждого тайла.
ACCEPTED_LEGACY_SOURCES: dict[str, float] = {}

CACHE_INDEX_VERSION = 3

# Меняется при изменении алгоритма чтения GeoTIFF.
GRID_READER_VERSION = 2


# ==================== ОШИБКИ ПАЙПЛАЙНА ====================

class TileValidationError(RuntimeError):
    """Тайл не соответствует требованиям или повреждён."""


# ==================== ПУТИ И JSON ====================

def resolve_path(path) -> Path:
    result = Path(path).expanduser()

    if not result.is_absolute():
        result = PROJECT_ROOT / result

    return result.resolve()


def bbox_dict(bbox):
    return {
        "south": float(bbox.south),
        "west": float(bbox.west),
        "north": float(bbox.north),
        "east": float(bbox.east),
    }


def covers(bounds, required):
    eps = 1e-10

    return (
        bounds["south"] <= required.south + eps
        and bounds["west"] <= required.west + eps
        and bounds["north"] >= required.north - eps
        and bounds["east"] >= required.east - eps
    )


def atomic_json_write(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        temporary.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path, default):
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return default


def fingerprint(path):
    stat = path.stat()

    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def provenance_path(path):
    return path.with_suffix(".provenance.json")


def recipe_hash(recipe):
    return hashlib.sha256(
        json.dumps(
            recipe,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


# ==================== API-КЛЮЧ И ЛОГИ ====================

def get_api_key():
    key = os.getenv(
        "OPENTOPOGRAPHY_API_KEY",
        "",
    ).strip()

    if key:
        return key

    try:
        module = importlib.import_module(
            "secret.OPEN_TOPOGRAPHY_API"
        )

    except ModuleNotFoundError as error:
        if error.name not in {
            "secret",
            "secret.OPEN_TOPOGRAPHY_API",
        }:
            raise

        return None

    key = getattr(
        module,
        "OP_TOP_KEY",
        None,
    )

    return str(key).strip() if key else None


def redact(value):
    text = str(value)

    text = re.sub(
        r"(?i)(API_Key=)[^&\s'\"<>]+",
        r"\1***",
        text,
    )

    key = os.getenv("OPENTOPOGRAPHY_API_KEY")

    if key:
        text = text.replace(key, "***")

    return text


def log_filter(record):
    record["message"] = redact(
        record["message"]
    )

    return True


def configure_logging(worker=False):
    logger.remove()

    logger.add(
        sys.stderr,
        level="WARNING" if worker else "INFO",
        filter=log_filter,
        diagnose=False,
        backtrace=False,
    )


# ==================== БЛОКИРОВКА КАТАЛОГА ====================

@contextmanager
def output_lock(output_dir):
    """
    Защита от двух одновременно запущенных копий.
    Для macOS/Linux используется flock.
    """
    import fcntl

    path = output_dir / ".download.lock"

    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

        except BlockingIOError as error:
            raise RuntimeError(
                "Другой процесс уже работает "
                f"с каталогом {output_dir}."
            ) from error

        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()

        try:
            yield

        finally:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_UN,
            )


# ==================== КОРРЕКТНОЕ ЧТЕНИЕ GEOTIFF ====================

def read_geotiff_grid(path, source_name):
    """
    Координаты вычисляются для центров пикселей.

    HTTP-загрузка остаётся в базовых загрузчиках.
    """
    import rasterio

    with rasterio.open(path) as src:
        if src.crs is None:
            raise TileValidationError(
                "У GeoTIFF отсутствует CRS."
            )

        if not CRS.from_user_input(src.crs).equals(
            CRS.from_epsg(4326),
            ignore_axis_order=True,
        ):
            raise TileValidationError(
                f"Ожидался WGS84, получено {src.crs}."
            )

        transform = src.transform

        if (
            abs(transform.b) > 1e-12
            or abs(transform.d) > 1e-12
        ):
            raise TileValidationError(
                "GeoTIFF имеет поворот: одномерные lat/lon "
                "не могут корректно описать такую сетку."
            )

        if (
            src.width < 2
            or src.height < 2
            or transform.a == 0
            or transform.e == 0
        ):
            raise TileValidationError(
                "Некорректные размеры или шаг GeoTIFF."
            )

        z = (
            src.read(1, masked=True)
            .astype(np.float64)
            .filled(np.nan)
        )

        lons = (
            transform.c
            + transform.a
            * (
                np.arange(
                    src.width,
                    dtype=np.float64,
                ) + 0.5
            )
        )

        lats = (
            transform.f
            + transform.e
            * (
                np.arange(
                    src.height,
                    dtype=np.float64,
                ) + 0.5
            )
        )

        if lons[0] > lons[-1]:
            lons = lons[::-1].copy()
            z = z[:, ::-1].copy()

        if lats[0] > lats[-1]:
            lats = lats[::-1].copy()
            z = z[::-1, :].copy()

        if not (
            np.isfinite(lons).all()
            and np.isfinite(lats).all()
            and np.all(np.diff(lons) > 0)
            and np.all(np.diff(lats) > 0)
        ):
            raise TileValidationError(
                "Некорректные координаты GeoTIFF."
            )

        resolution = float(
            max(
                abs(transform.a),
                abs(transform.e),
            ) * 3600.0
        )

        return BathymetryGrid(
            lats=lats,
            lons=lons,
            z=z,
            source=source_name,
            resolution_arcsec=resolution,
        )


class PipelineOpenTopographyLoader(
    GEBCOOpenTopographyLoader
):
    def _read_geotiff_as_grid(self, tif_path):
        return read_geotiff_grid(
            tif_path,
            self.source_name,
        )


class PipelineEMODnetLoader(
    EMODnetBathymetryLoader
):
    def _read_geotiff_as_grid(self, tif_path):
        return read_geotiff_grid(
            tif_path,
            self.source_name,
        )


# ==================== ВЫБОР БАЗОВОГО ЗАГРУЗЧИКА ====================

def loader_profile(bbox, config):
    use_emodnet = PROVIDER == "emodnet"

    if PROVIDER == "auto":
        selector = BathymetryArchiveBuilder(
            session=None,
            config=config,
        )

        selected = selector._default_loader(
            bbox
        )

        use_emodnet = isinstance(
            selected,
            EMODnetBathymetryLoader,
        )

        if not use_emodnet and not isinstance(
            selected,
            GEBCODownloadLoader,
        ):
            raise RuntimeError(
                "Неизвестный загрузчик в auto-режиме."
            )

    if use_emodnet:
        recipe = {
            "loader": "EMODnetBathymetryLoader",
            "base_url": EMODNET_URL,
            "coverage_id": EMODNET_COVERAGE,
            "grid_reader_version": GRID_READER_VERSION,
        }

        return {
            "kind": "emodnet",
            "source": "EMODnet_WCS",
            "resolution": EMODNET_MAX_RESOLUTION_ARCSEC,
            "recipe": recipe,
            "recipe_id": recipe_hash(recipe),
        }

    if GEBCO_BACKEND == "opentopography":
        recipe = {
            "loader": "GEBCOOpenTopographyLoader",
            "base_url": OPENTOPOGRAPHY_URL,
            "demtype": OPENTOPOGRAPHY_DEMTYPE,
            "grid_reader_version": GRID_READER_VERSION,
        }

        source = "GEBCO_OpenTopography"

    else:
        recipe = {
            "loader": "GEBCOOpenDAPLoader",
            "opendap_url": GEBCO_OPENDAP_URL,
            "engine": GEBCO_OPENDAP_ENGINE,
        }

        source = "GEBCO_OPeNDAP"

    return {
        "kind": GEBCO_BACKEND,
        "source": source,
        "resolution": GEBCO_MAX_RESOLUTION_ARCSEC,
        "recipe": recipe,
        "recipe_id": recipe_hash(recipe),
    }


def create_loader(profile):
    if profile["kind"] == "emodnet":
        return PipelineEMODnetLoader(
            base_url=EMODNET_URL,
            coverage_id=EMODNET_COVERAGE,
            timeout=PROVIDER_TIMEOUT_S,
            save_download=False,
        )

    if profile["kind"] == "opentopography":
        return PipelineOpenTopographyLoader(
            api_key=get_api_key(),
            base_url=OPENTOPOGRAPHY_URL,
            demtype=OPENTOPOGRAPHY_DEMTYPE,
            timeout=PROVIDER_TIMEOUT_S,
            save_download=False,
            require_api_key=True,
        )

    return GEBCOOpenDAPLoader(
        opendap_url=GEBCO_OPENDAP_URL,
        engine=GEBCO_OPENDAP_ENGINE,
    )


# ==================== ПРОВЕРКА NETCDF ====================

def inspect_netcdf(path):
    with xr.open_dataset(
        path,
        decode_times=False,
        cache=False,
    ) as dataset:
        if not {
            "lat",
            "lon",
            "elevation",
        }.issubset(dataset.variables):
            raise TileValidationError(
                "В NetCDF отсутствуют lat/lon/elevation."
            )

        lats = np.asarray(
            dataset["lat"].values,
            dtype=float,
        )

        lons = np.asarray(
            dataset["lon"].values,
            dtype=float,
        )

        z = dataset["elevation"]

        if (
            lats.ndim != 1
            or lons.ndim != 1
            or len(lats) < 2
            or len(lons) < 2
            or z.dims != ("lat", "lon")
            or z.shape != (len(lats), len(lons))
            or not np.isfinite(lats).all()
            or not np.isfinite(lons).all()
        ):
            raise TileValidationError(
                "Некорректная структура NetCDF."
            )

        dlat = np.diff(lats)
        dlon = np.diff(lons)

        if not (
            (
                np.all(dlat > 0)
                or np.all(dlat < 0)
            )
            and (
                np.all(dlon > 0)
                or np.all(dlon < 0)
            )
        ):
            raise TileValidationError(
                "Координатные оси не монотонны."
            )

        if (
            lats.min() < -90.0
            or lats.max() > 90.0
            or lons.min() < -180.0
            or lons.max() > 180.0
        ):
            raise TileValidationError(
                "Координаты NetCDF вне диапазонов WGS84."
            )

        block_rows = max(
            1,
            VERIFY_BLOCK_CELLS // len(lons),
        )

        finite_count = 0

        for start in range(
            0,
            len(lats),
            block_rows,
        ):
            block = np.asarray(
                z.isel(
                    lat=slice(
                        start,
                        start + block_rows,
                    )
                ).values
            )

            finite_count += int(
                np.isfinite(block).sum()
            )

        if finite_count == 0:
            raise TileValidationError(
                "Матрица глубин полностью пуста."
            )

        metadata = {
            "path": str(path.resolve()),
            "grid_bbox": {
                "south": float(lats.min()),
                "west": float(lons.min()),
                "north": float(lats.max()),
                "east": float(lons.max()),
            },
            "shape": [
                len(lats),
                len(lons),
            ],
            "source": str(
                dataset.attrs.get(
                    "source",
                    z.attrs.get("source", ""),
                )
            ),
            "resolution_arcsec": float(
                max(
                    np.max(np.abs(dlat)),
                    np.max(np.abs(dlon)),
                ) * 3600.0
            ),
            "full_data_checked": True,
            "finite_count": finite_count,
        }

    metadata["fingerprint"] = fingerprint(path)

    return metadata


def attach_provenance(metadata, path):
    sidecar = read_json(
        provenance_path(path),
        {},
    )

    if sidecar.get("fingerprint") == metadata["fingerprint"]:
        metadata["recipe"] = sidecar.get("recipe")
        metadata["recipe_id"] = sidecar.get("recipe_id")

    else:
        metadata["recipe"] = None
        metadata["recipe_id"] = None

    return metadata


def scan_cache(output_dir):
    index_path = output_dir / "cache_index.json"
    previous = read_json(index_path, {})

    old_files = (
        previous.get("files", {})
        if previous.get("version") == CACHE_INDEX_VERSION
        else {}
    )

    roots = [
        output_dir,
        *[
            resolve_path(path)
            for path in ADDITIONAL_CACHE_DIRS
        ],
    ]

    paths = set()

    for root in roots:
        if not root.is_dir():
            continue

        paths.update(
            path.resolve()
            for path in root.rglob("*.nc")
            if ".part." not in path.name
            and ".tmp." not in path.name
        )

    cache = {}

    for path in tqdm(
        sorted(paths),
        desc="Проверка кеша",
        unit="файл",
        dynamic_ncols=True,
    ):
        try:
            old = old_files.get(str(path))

            if (
                not VERIFY_ALL_CACHE_DATA_EVERY_RUN
                and isinstance(old, dict)
                and old.get("fingerprint") == fingerprint(path)
                and old.get("full_data_checked") is True
            ):
                metadata = dict(old)

            else:
                metadata = inspect_netcdf(path)

            metadata = attach_provenance(
                metadata,
                path,
            )

            cache[str(path)] = metadata

        except Exception as error:
            logger.warning(
                "Файл не принят в кеш: {}: {}",
                path,
                redact(error),
            )

    atomic_json_write(
        {
            "version": CACHE_INDEX_VERSION,
            "files": cache,
        },
        index_path,
    )

    logger.info(
        "Пригодных для последующего подбора файлов: {}",
        len(cache),
    )

    return cache


# ==================== ПОДБОР КЕША ====================

def compatible(metadata, profile):
    tolerance = RESOLUTION_TOLERANCE_ARCSEC

    if metadata.get("recipe_id") == profile["recipe_id"]:
        return (
            metadata["source"] == profile["source"]
            and metadata["resolution_arcsec"]
            <= profile["resolution"] + tolerance
        )

    if metadata.get("recipe_id") is None:
        legacy_limit = ACCEPTED_LEGACY_SOURCES.get(
            metadata["source"]
        )

        return (
            legacy_limit is not None
            and metadata["resolution_arcsec"]
            <= min(
                legacy_limit,
                profile["resolution"],
            ) + tolerance
        )

    return False


def find_cached(required, profile, cache):
    candidates = [
        metadata
        for metadata in cache.values()
        if compatible(metadata, profile)
        and covers(
            metadata["grid_bbox"],
            required,
        )
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: (
            item["shape"][0] * item["shape"][1],
            item["path"],
        ),
    )


def tile_id_for(path):
    digest = hashlib.sha256(
        str(Path(path).resolve()).encode("utf-8")
    ).hexdigest()[:24]

    return f"tile_{digest}"


def attach_points(assignments, metadata, points):
    tile_id = tile_id_for(metadata["path"])

    entry = assignments.setdefault(
        tile_id,
        {
            "metadata": metadata,
            "points": {},
        },
    )

    for point in points:
        entry["points"][point.point_id] = point


# ==================== ПЛАН НЕДОСТАЮЩИХ ДАННЫХ ====================

def prepare_jobs(points, cache, config):
    assignments = {}
    missing = []

    for point in tqdm(
        points,
        desc="Подбор готового покрытия",
        unit="точ.",
        dynamic_ncols=True,
    ):
        request = _expand_bbox(
            point.required_bbox,
            config.padding_m,
        )

        profile = loader_profile(
            request,
            config,
        )

        metadata = find_cached(
            point.required_bbox,
            profile,
            cache,
        )

        if metadata is None:
            missing.append(point)

        else:
            attach_points(
                assignments,
                metadata,
                [point],
            )

    logger.info(
        "Точек: {}; покрыто кешем: {}; без покрытия: {}",
        len(points),
        len(points) - len(missing),
        len(missing),
    )

    missing.sort(
        key=lambda point: (
            math.floor((point.lon + 180.0) / 2.0),
            math.floor((point.lat + 90.0) / 2.0),
            point.lon,
            point.lat,
            point.point_id,
        )
    )

    jobs = []
    start = 0

    while start < len(missing):
        index = start
        chunk = []
        current_bbox = None
        current_profile = None

        while index < len(missing):
            point = missing[index]

            candidate = (
                point.required_bbox
                if current_bbox is None
                else _union_bbox(
                    (
                        current_bbox,
                        point.required_bbox,
                    )
                )
            )

            request = _expand_bbox(
                candidate,
                config.padding_m,
            )

            profile = loader_profile(
                request,
                config,
            )

            cells = _estimate_cells(
                request,
                config.planning_resolution_arcsec,
            )

            split = (
                cells > config.max_cells_per_tile
                or request.north - request.south
                > MAX_LAT_SPAN_DEG
                or request.east - request.west
                > MAX_LON_SPAN_DEG
                or (
                    current_profile is not None
                    and current_profile["recipe_id"]
                    != profile["recipe_id"]
                )
            )

            if chunk and split:
                break

            if not chunk and cells > config.max_cells_per_tile:
                raise ValueError(
                    f"Точка {point.point_id} требует "
                    f"примерно {cells:,} ячеек при лимите "
                    f"{config.max_cells_per_tile:,}."
                )

            chunk.append(point)
            current_bbox = candidate
            current_profile = profile
            index += 1

        request = _expand_bbox(
            current_bbox,
            config.padding_m,
        )

        profile = loader_profile(
            request,
            config,
        )

        cached = find_cached(
            current_bbox,
            profile,
            cache,
        )

        if cached is not None:
            attach_points(
                assignments,
                cached,
                chunk,
            )

        else:
            digest = recipe_hash(
                {
                    "recipe_id": profile["recipe_id"],
                    "request_bbox": bbox_dict(request),
                }
            )[:32]

            jobs.append(
                {
                    "id": f"download_{digest}",
                    "required": current_bbox,
                    "request": request,
                    "profile": profile,
                    "filename": f"tile_{digest}.nc",
                    "points": chunk,
                }
            )

        start = index

    return assignments, jobs


# ==================== ДИАГНОСТИКА ТАЙЛА ====================

def validate_tile_metadata(metadata, job):
    profile = job["profile"]
    problems = []

    if metadata["source"] != profile["source"]:
        problems.append(
            "источник: "
            f"получен {metadata['source']!r}, "
            f"ожидался {profile['source']!r}"
        )

    if metadata["resolution_arcsec"] > (
        profile["resolution"]
        + RESOLUTION_TOLERANCE_ARCSEC
    ):
        problems.append(
            "разрешение: "
            f"получено {metadata['resolution_arcsec']:.9f}″, "
            f"допустимо {profile['resolution']:.9f}″"
        )

    if not covers(
        metadata["grid_bbox"],
        job["required"],
    ):
        problems.append(
            "покрытие: "
            f"grid={metadata['grid_bbox']}, "
            f"required={bbox_dict(job['required'])}"
        )

    if problems:
        raise TileValidationError(
            f"{job['id']}: "
            + "; ".join(problems)
            + f"; shape={metadata['shape']}"
        )


# ==================== ПОЛИТИКА ПОВТОРОВ ====================

def exception_chain(error):
    seen = set()

    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def parse_retry_after(value):
    if not value:
        return None

    try:
        seconds = float(value)

        if math.isfinite(seconds):
            return max(0.0, seconds)

    except ValueError:
        pass

    try:
        timestamp = parsedate_to_datetime(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        return max(
            0.0,
            (
                timestamp - datetime.now(UTC)
            ).total_seconds(),
        )

    except (TypeError, ValueError, OverflowError):
        return None


def retry_information(error):
    status = None
    retry_after = None

    for item in exception_chain(error):
        if isinstance(
            item,
            (
                TileValidationError,
                BathymetryAuthenticationError,
                BathymetryConfigurationError,
                ImportError,
            ),
        ):
            return False, None

        item_status = getattr(
            item,
            "status_code",
            None,
        )

        if item_status is not None:
            status = item_status

        response = getattr(
            item,
            "response",
            None,
        )

        if response is not None:
            status = response.status_code

            retry_after = parse_retry_after(
                response.headers.get("Retry-After")
            )

    if status is not None:
        return (
            status in {408, 429, 500, 502, 503, 504},
            retry_after,
        )

    # Ошибки файловой системы не исправляются повторным HTTP.
    if isinstance(error, OSError):
        return False, None

    return True, retry_after


# ==================== РАБОЧИЙ ПРОЦЕСС ====================

def download_worker(job, output_dir_string):
    configure_logging(worker=True)

    output_dir = Path(output_dir_string)
    path = output_dir / "tiles" / job["filename"]

    temporary = path.with_suffix(".part.nc")
    rejected = path.with_suffix(".rejected")
    report = path.with_suffix(".validation.json")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = job["profile"]

    if path.is_file():
        try:
            metadata = attach_provenance(
                inspect_netcdf(path),
                path,
            )

            if compatible(metadata, profile):
                validate_tile_metadata(
                    metadata,
                    job,
                )

                return metadata

        except Exception:
            pass

    last_error = None

    for attempt in range(
        1,
        DOWNLOAD_ATTEMPTS + 1,
    ):
        temporary.unlink(missing_ok=True)

        try:
            loader = create_loader(profile)

            # Сеть и получение исходного растра —
            # реализация базового проекта.
            grid = loader.load(
                job["request"]
            )

            NetCDFBathymetryExporter().export(
                grid,
                temporary,
            )

            del grid

            try:
                metadata = inspect_netcdf(
                    temporary
                )

                validate_tile_metadata(
                    metadata,
                    job,
                )

            except Exception as error:
                # Сохраняем проблемный ответ отдельно.
                # Сканер кеша не ищет расширение .rejected.
                if temporary.is_file():
                    temporary.replace(rejected)

                atomic_json_write(
                    {
                        "job": job["id"],
                        "error": redact(error),
                        "profile": profile,
                        "required_bbox": bbox_dict(
                            job["required"]
                        ),
                        "request_bbox": bbox_dict(
                            job["request"]
                        ),
                        "rejected_file": str(rejected),
                        "actual_metadata": locals().get(
                            "metadata"
                        ),
                    },
                    report,
                )

                raise TileValidationError(
                    redact(error)
                ) from error

            temporary.replace(path)

            metadata["path"] = str(path.resolve())
            metadata["fingerprint"] = fingerprint(path)
            metadata["recipe"] = profile["recipe"]
            metadata["recipe_id"] = profile["recipe_id"]

            atomic_json_write(
                {
                    "fingerprint": metadata["fingerprint"],
                    "recipe": profile["recipe"],
                    "recipe_id": profile["recipe_id"],
                },
                provenance_path(path),
            )

            rejected.unlink(missing_ok=True)
            report.unlink(missing_ok=True)

            return metadata

        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)

            retryable, retry_after = retry_information(
                error
            )

            if (
                not retryable
                or attempt == DOWNLOAD_ATTEMPTS
            ):
                break

            delay = max(
                min(
                    RETRY_MAX_S,
                    RETRY_INITIAL_S
                    * (2 ** (attempt - 1)),
                ),
                retry_after or 0.0,
            )

            delay += random.uniform(
                0.0,
                RETRY_JITTER_S,
            )

            logger.warning(
                "{}: попытка {}/{}: {}. "
                "Повтор через {:.1f} с.",
                job["id"],
                attempt,
                DOWNLOAD_ATTEMPTS,
                redact(error),
                delay,
            )

            time.sleep(delay)

    raise RuntimeError(
        f"{job['id']}: {redact(last_error)}"
    ) from None


# ==================== ОЧЕРЕДЬ ЗАГРУЗКИ ====================

def download_jobs(
    jobs,
    output_dir,
    cache,
    assignments,
):
    if not jobs:
        logger.success(
            "Все текущие точки покрыты кешем. "
            "Новые загрузки не нужны."
        )
        return

    if any(
        job["profile"]["kind"] == "opentopography"
        for job in jobs
    ):
        if not get_api_key():
            raise ValueError(
                "Для новых загрузок OpenTopography нужен "
                "OPENTOPOGRAPHY_API_KEY либо "
                "secret/OPEN_TOPOGRAPHY_API.py с OP_TOP_KEY."
            )

    failures = []
    pending = {}
    iterator = iter(jobs)

    def save_state():
        atomic_json_write(
            {
                "version": CACHE_INDEX_VERSION,
                "files": cache,
            },
            output_dir / "cache_index.json",
        )

        atomic_json_write(
            {
                "status": "incomplete",
                "updated_at": datetime.now(UTC).isoformat(),
                "failures": failures,
                "verified_files": len(cache),
            },
            output_dir / "manifest.partial.json",
        )

    def submit_next(executor):
        try:
            job = next(iterator)

        except StopIteration:
            return

        future = executor.submit(
            download_worker,
            job,
            str(output_dir),
        )

        pending[future] = job

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        for _ in range(
            min(
                WORKERS * QUEUE_MULTIPLIER,
                len(jobs),
            )
        ):
            submit_next(executor)

        with tqdm(
            total=len(jobs),
            desc="Загрузка батиметрии",
            unit="тайл",
            dynamic_ncols=True,
        ) as progress:
            while pending:
                done, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    job = pending.pop(future)

                    try:
                        metadata = future.result()

                        cache[metadata["path"]] = metadata

                        attach_points(
                            assignments,
                            metadata,
                            job["points"],
                        )

                    except Exception as error:
                        failures.append(
                            {
                                "job": job["id"],
                                "point_ids": [
                                    point.point_id
                                    for point in job["points"]
                                ],
                                "error": redact(error),
                            }
                        )

                        logger.error(
                            "{}",
                            redact(error),
                        )

                    save_state()
                    progress.update(1)

                    progress.set_postfix(
                        failed=len(failures),
                        refresh=False,
                    )

                    submit_next(executor)

    if failures:
        raise RuntimeError(
            f"Не загружено тайлов: {len(failures)}. "
            "Готовые файлы сохранены для повторного запуска. "
            "Итоговый manifest.json не заменён. "
            "Подробности: manifest.partial.json и "
            "*.validation.json."
        )


# ==================== ИТОГОВЫЙ МАНИФЕСТ ====================

def write_manifest(
    source_id,
    points,
    assignments,
    config,
):
    output_dir = config.output_dir
    point_to_tiles = {}
    tiles = []

    for tile_id, entry in sorted(assignments.items()):
        metadata = entry["metadata"]
        assigned = list(entry["points"].values())
        path = Path(metadata["path"])

        if fingerprint(path) != metadata["fingerprint"]:
            raise RuntimeError(
                f"Файл изменился во время выполнения: {path}"
            )

        for point in assigned:
            if not covers(
                metadata["grid_bbox"],
                point.required_bbox,
            ):
                raise RuntimeError(
                    f"Нет покрытия point_id={point.point_id}."
                )

            if point.point_id in point_to_tiles:
                raise RuntimeError(
                    "Повторное назначение точки."
                )

            point_to_tiles[point.point_id] = [tile_id]

        required = _union_bbox(
            point.required_bbox
            for point in assigned
        )

        tiles.append(
            {
                "tile_id": tile_id,
                "file": Path(
                    os.path.relpath(path, output_dir)
                ).as_posix(),
                "point_ids": [
                    point.point_id
                    for point in assigned
                ],
                "seq_min": min(
                    point.seq for point in assigned
                ),
                "seq_max": max(
                    point.seq for point in assigned
                ),
                "max_fetch_m": max(
                    point.max_fetch_m
                    for point in assigned
                ),
                "required_bbox": bbox_dict(required),
                # Для ранее скачанного тайла исходный
                # HTTP bbox может быть неизвестен.
                "request_bbox": metadata["grid_bbox"],
                "grid_bbox": metadata["grid_bbox"],
                "estimated_cells": int(
                    metadata["shape"][0]
                    * metadata["shape"][1]
                ),
                "shape": metadata["shape"],
                "source": metadata["source"],
                "resolution_arcsec": (
                    metadata["resolution_arcsec"]
                ),
                "recipe": metadata.get("recipe"),
            }
        )

    expected_ids = {
        point.point_id
        for point in points
    }

    if set(point_to_tiles) != expected_ids:
        missing = sorted(
            expected_ids - set(point_to_tiles)
        )

        raise RuntimeError(
            "Не все текущие точки покрыты архивом. "
            f"Примеры id: {missing[:20]}"
        )

    config_data = asdict(config)
    config_data["output_dir"] = str(output_dir)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "point_source_id": source_id,
        "config": config_data,
        "pipeline_metadata": {
            "cache_policy": (
                "coverage_recipe_resolution"
            ),
            "grid_reader_version": GRID_READER_VERSION,
        },
        "points": {
            str(point.point_id): {
                "seq": point.seq,
                "lon": point.lon,
                "lat": point.lat,
                "max_fetch_m": point.max_fetch_m,
                "tile_ids": point_to_tiles[point.point_id],
            }
            for point in points
        },
        "tiles": tiles,
    }

    result = output_dir / "manifest.json"

    atomic_json_write(
        manifest,
        result,
    )

    (
        output_dir / "manifest.partial.json"
    ).unlink(missing_ok=True)

    return result


# ==================== ПРОВЕРКА НАСТРОЕК ====================

def validate_settings():
    if PROVIDER not in {"gebco", "emodnet", "auto"}:
        raise ValueError(
            "PROVIDER: gebco | emodnet | auto."
        )

    if GEBCO_BACKEND not in {"opentopography", "opendap"}:
        raise ValueError(
            "GEBCO_BACKEND: opentopography | opendap."
        )

    for name, value in (
        ("WORKERS", WORKERS),
        ("QUEUE_MULTIPLIER", QUEUE_MULTIPLIER),
        ("DOWNLOAD_ATTEMPTS", DOWNLOAD_ATTEMPTS),
        ("VERIFY_BLOCK_CELLS", VERIFY_BLOCK_CELLS),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"{name} должен быть целым >= 1."
            )

    for name, value in (
        ("SAFETY_FACTOR", SAFETY_FACTOR),
        ("PADDING_M", PADDING_M),
        (
            "PLANNING_RESOLUTION_ARCSEC",
            PLANNING_RESOLUTION_ARCSEC,
        ),
        ("MAX_LAT_SPAN_DEG", MAX_LAT_SPAN_DEG),
        ("MAX_LON_SPAN_DEG", MAX_LON_SPAN_DEG),
        ("RETRY_INITIAL_S", RETRY_INITIAL_S),
        ("RETRY_MAX_S", RETRY_MAX_S),
        (
            "GEBCO_MAX_RESOLUTION_ARCSEC",
            GEBCO_MAX_RESOLUTION_ARCSEC,
        ),
        (
            "EMODNET_MAX_RESOLUTION_ARCSEC",
            EMODNET_MAX_RESOLUTION_ARCSEC,
        ),
    ):
        if not math.isfinite(value):
            raise ValueError(
                f"{name} должен быть конечным."
            )

    if not math.isfinite(RETRY_JITTER_S) or RETRY_JITTER_S < 0:
        raise ValueError(
            "RETRY_JITTER_S должен быть >= 0."
        )

    if (
        not math.isfinite(RESOLUTION_TOLERANCE_ARCSEC)
        or RESOLUTION_TOLERANCE_ARCSEC < 0
    ):
        raise ValueError(
            "Некорректный допуск разрешения."
        )

    for value in (
        MAX_LAT_SPAN_DEG,
        MAX_LON_SPAN_DEG,
        RETRY_INITIAL_S,
        RETRY_MAX_S,
        GEBCO_MAX_RESOLUTION_ARCSEC,
        EMODNET_MAX_RESOLUTION_ARCSEC,
    ):
        if value <= 0:
            raise ValueError(
                "Лимиты размеров, задержки и разрешения "
                "должны быть положительными."
            )

    for source, resolution in ACCEPTED_LEGACY_SOURCES.items():
        if (
            not isinstance(source, str)
            or not math.isfinite(resolution)
            or resolution <= 0
        ):
            raise ValueError(
                "Некорректный ACCEPTED_LEGACY_SOURCES."
            )


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

def run_pipeline(output_dir):
    database_path = resolve_path(
        DATABASE_PATH
    )

    if not database_path.is_file():
        raise FileNotFoundError(
            f"База не найдена:\n{database_path}"
        )

    config = BathymetryArchiveConfig(
        output_dir=output_dir,
        provider=PROVIDER,
        safety_factor=SAFETY_FACTOR,
        padding_m=PADDING_M,
        max_cells_per_tile=MAX_CELLS_PER_TILE,
        planning_resolution_arcsec=(
            PLANNING_RESOLUTION_ARCSEC
        ),
        overlap_points=0,
        retries=1,
        force=False,
        max_fetch_override_m=MAX_FETCH_M,
    )

    logger.info("База: {}", database_path)
    logger.info("Архив: {}", output_dir)

    logger.info(
        "PROVIDER={}, GEBCO_BACKEND={}",
        PROVIDER,
        GEBCO_BACKEND,
    )

    if ACCEPTED_LEGACY_SOURCES:
        logger.warning(
            "Явно разрешены прежние источники: {}",
            list(ACCEPTED_LEGACY_SOURCES),
        )

    cache = scan_cache(
        output_dir
    )

    engine = create_engine(
        URL.create(
            "sqlite",
            database=str(database_path),
        ),
        connect_args={"timeout": 60},
    )

    try:
        with Session(engine) as session:
            session.connection().exec_driver_sql(
                "PRAGMA query_only = ON"
            )

            planner = BathymetryCoveragePlanner(
                session=session,
                config=config,
            )

            source_id = planner.resolve_point_source_id(
                POINT_SOURCE_ID
            )

            logger.info(
                "Чтение актуальных точек и fetch..."
            )

            points = planner.load_point_coverages(
                source_id
            )

    finally:
        engine.dispose()

    if not points:
        raise ValueError(
            "Нет точек для обработки."
        )

    assignments, jobs = prepare_jobs(
        points,
        cache,
        config,
    )

    logger.info(
        "Новых запросов: {}; процессов: {}",
        len(jobs),
        WORKERS,
    )

    download_jobs(
        jobs,
        output_dir,
        cache,
        assignments,
    )

    manifest = write_manifest(
        source_id,
        points,
        assignments,
        config,
    )

    logger.success(
        "Архив готов: {} точек, {} файлов. {}",
        len(points),
        len(assignments),
        manifest,
    )


def main():
    validate_settings()

    key = get_api_key()

    if key:
        # Рабочие процессы spawn наследуют окружение.
        os.environ["OPENTOPOGRAPHY_API_KEY"] = key

    configure_logging()

    output_dir = resolve_path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_lock(output_dir):
        run_pipeline(output_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()