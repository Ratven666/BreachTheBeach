from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import requests
import rasterio

from loguru import logger
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin

from .logging_utils import stage


EMODNET_URL = "https://ows.emodnet-bathymetry.eu/wcs"
EMODNET_COVERAGE = "emodnet:mean"
VERSION = "dalain-bathymetry-2"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def raster_valid(path):
    if not path.is_file():
        return False

    try:
        with rasterio.open(path) as src:
            return (
                src.count == 1
                and src.width > 1
                and src.height > 1
                and src.crs is not None
                and src.crs.to_epsg() == 4326
                and src.dtypes[0] == "float32"
                and src.transform.a > 0
                and src.transform.e < 0
                and src.transform.b == 0
                and src.transform.d == 0
            )
    except Exception:
        return False


def coverage(cfg):
    extent = (
        max(cfg.search_radius_m, cfg.evaluation_radius_m)
        + cfg.max_fetch_m * cfg.bathymetry_safety_factor
        + cfg.bathymetry_padding_m
    )

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={cfg.center_lat} "
        f"+lon_0={cfg.center_lon} +datum=WGS84 +units=m"
    )
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)

    values = np.linspace(-extent, extent, 513)

    xs = np.concatenate([
        values, values,
        np.full_like(values, -extent),
        np.full_like(values, extent),
    ])
    ys = np.concatenate([
        np.full_like(values, -extent),
        np.full_like(values, extent),
        values, values,
    ])

    lons, lats = inverse.transform(xs, ys)

    if not (np.isfinite(lons).all() and np.isfinite(lats).all()):
        raise ValueError("Не удалось рассчитать покрытие батиметрии")

    if np.ptp(lons) > 180 or np.min(lats) <= -89 or np.max(lats) >= 89:
        raise ValueError(
            "Антимеридиан и полярные области требуют отдельного планировщика"
        )

    margin = 2 * cfg.bathymetry_planning_resolution_arcsec / 3600

    return {
        "west": float(np.min(lons) - margin),
        "east": float(np.max(lons) + margin),
        "south": float(np.min(lats) - margin),
        "north": float(np.max(lats) + margin),
    }


def tiles_for(bounds, cfg):
    resolution = cfg.bathymetry_planning_resolution_arcsec / 3600
    overlap = cfg.bathymetry_tile_overlap_cells * resolution

    pixels = math.floor(math.sqrt(cfg.bathymetry_max_cells_per_tile))
    core_pixels = pixels - 2 * cfg.bathymetry_tile_overlap_cells - 4

    if core_pixels < 2:
        raise ValueError("Слишком мал bathymetry_max_cells_per_tile")

    span = core_pixels * resolution

    nx = max(1, math.ceil((bounds["east"] - bounds["west"]) / span))
    ny = max(1, math.ceil((bounds["north"] - bounds["south"]) / span))

    longitudes = np.linspace(bounds["west"], bounds["east"], nx + 1)
    latitudes = np.linspace(bounds["south"], bounds["north"], ny + 1)

    result = []

    for j in range(ny):
        for i in range(nx):
            result.append({
                "west": max(bounds["west"], float(longitudes[i]) - overlap),
                "east": min(bounds["east"], float(longitudes[i + 1]) + overlap),
                "south": max(bounds["south"], float(latitudes[j]) - overlap),
                "north": min(bounds["north"], float(latitudes[j + 1]) + overlap),
            })

    return result


def profile_for(width, height, affine):
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": affine,
        "nodata": float("nan"),
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }


def fetch_emodnet(bounds, destination, cfg):
    raw = destination.with_suffix(".download.tif")

    params = [
        ("SERVICE", "WCS"),
        ("REQUEST", "GetCoverage"),
        ("VERSION", "2.0.1"),
        ("COVERAGEID", EMODNET_COVERAGE),
        ("FORMAT", "image/tiff"),
        ("SUBSET", f"Long({bounds['west']},{bounds['east']})"),
        ("SUBSET", f"Lat({bounds['south']},{bounds['north']})"),
    ]

    logger.debug("EMODnet запрос: {}", params)

    try:
        with requests.get(
            EMODNET_URL,
            params=params,
            stream=True,
            timeout=cfg.bathymetry_download_timeout_s,
        ) as response:
            response.raise_for_status()

            downloaded = 0
            last_log = time.perf_counter()

            with raw.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue

                    handle.write(chunk)
                    downloaded += len(chunk)

                    now = time.perf_counter()

                    if now - last_log >= cfg.progress_interval_s:
                        logger.info(
                            "EMODnet: получено {:.1f} МиБ",
                            downloaded / 1024**2,
                        )
                        last_log = now

            logger.info(
                "EMODnet: загрузка завершена, {:.1f} МиБ",
                downloaded / 1024**2,
            )

        with rasterio.open(raw) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError("EMODnet вернул неожиданную CRS")

            if (
                src.transform.a <= 0 or src.transform.e >= 0
                or src.transform.b or src.transform.d
            ):
                raise ValueError("Неожиданная геопривязка EMODnet")

            logger.debug(
                "EMODnet raster: {}x{}, resolution={}",
                src.width, src.height, src.res,
            )

            with rasterio.open(
                destination, "w",
                **profile_for(src.width, src.height, src.transform),
            ) as dst:
                for _, window in src.block_windows(1):
                    values = (
                        src.read(1, window=window, masked=True)
                        .astype("float32")
                        .filled(np.nan)
                    )
                    values *= cfg.bathymetry_emodnet_sign
                    dst.write(values, 1, window=window)

    finally:
        raw.unlink(missing_ok=True)


def fetch_gebco(bounds, destination, cfg):
    # Используем существующий загрузчик репозитория.
    from src.base.BBox import BBox
    from src.bathymetry.loaders.GEBCODownloadLoader import (
        GEBCODownloadLoader,
    )

    grid = GEBCODownloadLoader(
        timeout=cfg.bathymetry_download_timeout_s
    ).load(BBox(**bounds))

    iy = np.argsort(grid.lats)[::-1]
    ix = np.argsort(grid.lons)

    latitudes = np.asarray(grid.lats, dtype=float)[iy]
    longitudes = np.asarray(grid.lons, dtype=float)[ix]
    values = np.asarray(grid.z, dtype="float32")[np.ix_(iy, ix)]

    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError("GEBCO вернул слишком маленькую сетку")

    dx = float(np.median(np.diff(longitudes)))
    dy = float(-np.median(np.diff(latitudes)))

    if dx <= 0 or dy <= 0:
        raise ValueError("Некорректный шаг GEBCO")

    if not (
        np.allclose(np.diff(longitudes), dx, rtol=1e-5, atol=1e-10)
        and np.allclose(-np.diff(latitudes), dy, rtol=1e-5, atol=1e-10)
    ):
        raise ValueError("Нерегулярная сетка GEBCO")

    affine = from_origin(
        longitudes[0] - dx / 2,
        latitudes[0] + dy / 2,
        dx, dy,
    )

    with rasterio.open(
        destination, "w",
        **profile_for(len(longitudes), len(latitudes), affine),
    ) as dst:
        dst.write(values, 1)


def fetch_tile(bounds, destination, cfg):
    providers = (
        ["emodnet", "gebco"]
        if cfg.bathymetry_provider == "auto"
        else [cfg.bathymetry_provider]
    )

    errors = []
    temporary = destination.with_suffix(".partial.tif")

    for provider in providers:
        for attempt in range(1, cfg.bathymetry_download_retries + 1):
            temporary.unlink(missing_ok=True)
            started = time.perf_counter()

            try:
                logger.info(
                    "Загрузка {} | попытка {}/{}",
                    provider, attempt, cfg.bathymetry_download_retries,
                )

                if provider == "emodnet":
                    fetch_emodnet(bounds, temporary, cfg)
                else:
                    fetch_gebco(bounds, temporary, cfg)

                if not raster_valid(temporary):
                    raise ValueError("Тайл не прошёл проверку")

                temporary.replace(destination)

                logger.success(
                    "Тайл сохранён | {} | {:.1f} с",
                    destination.name, time.perf_counter() - started,
                )

                return {
                    "provider": provider,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }

            except Exception as error:
                temporary.unlink(missing_ok=True)
                message = f"{provider}: {type(error).__name__}: {error}"
                errors.append(message)
                logger.warning("Ошибка загрузки | {}", message)
                logger.opt(exception=True).debug("Детали ошибки загрузки")

                if attempt < cfg.bathymetry_download_retries:
                    time.sleep(min(30, 2**attempt))

        if provider != providers[-1]:
            logger.warning("Переход к резервному источнику батиметрии")

    raise RuntimeError(
        "Не удалось загрузить тайл:\n" + "\n".join(errors)
    )


def verify_negative_depths(path):
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            values = (
                src.read(1, window=window, masked=True)
                .astype(float)
                .filled(np.nan)
            )

            if (np.isfinite(values) & (values < 0)).any():
                return

    raise ValueError(
        "В растре нет отрицательных высот дна. "
        "Проверьте область и bathymetry_emodnet_sign."
    )


def prepare_bathymetry(cfg):
    if cfg.bathymetry_path is not None:
        logger.info("Локальная батиметрия: {}", cfg.bathymetry_path)

        with rasterio.open(cfg.bathymetry_path) as src:
            if src.crs is None:
                raise ValueError("У локальной батиметрии не задана CRS")

            logger.info(
                "Растр {}x{} | CRS={} | шаг={}",
                src.width, src.height, src.crs, src.res,
            )

        verify_negative_depths(cfg.bathymetry_path)
        return cfg

    if not cfg.bathymetry_auto_download:
        if cfg.bathymetry_required:
            raise ValueError("Реальная батиметрия обязательна")

        logger.warning(
            "Батиметрия отключена: используются постоянные глубины"
        )
        return cfg

    bounds = coverage(cfg)

    specification = {
        "version": VERSION,
        "bounds": bounds,
        "provider": cfg.bathymetry_provider,
        "resolution_arcsec": cfg.bathymetry_planning_resolution_arcsec,
        "max_cells": cfg.bathymetry_max_cells_per_tile,
        "overlap": cfg.bathymetry_tile_overlap_cells,
        "emodnet_sign": cfg.bathymetry_emodnet_sign,
        "emodnet_url": EMODNET_URL,
        "coverage": EMODNET_COVERAGE,
    }

    signature = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()[:20]

    directory = cfg.bathymetry_cache_directory / signature
    directory.mkdir(parents=True, exist_ok=True)

    manifest_path = directory / "manifest.json"
    mosaic = directory / "bathymetry.tif"

    old = {}

    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("Повреждён manifest; кеш будет перепроверен")

    if (
        not cfg.bathymetry_force_download
        and old.get("specification") == specification
        and old.get("complete") is True
        and raster_valid(mosaic)
    ):
        logger.success("Батиметрия найдена в кеше: {}", mosaic)
        return replace(cfg, bathymetry_path=mosaic)

    tiles = tiles_for(bounds, cfg)

    logger.info(
        "Покрытие W={:.5f}, E={:.5f}, S={:.5f}, N={:.5f}",
        bounds["west"], bounds["east"],
        bounds["south"], bounds["north"],
    )
    logger.info(
        "Тайлов: {} | источник: {} | кеш: {}",
        len(tiles), cfg.bathymetry_provider, directory,
    )

    previous = (
        {item["file"]: item for item in old.get("tiles", [])}
        if old.get("specification") == specification else {}
    )

    manifest = {
        "specification": specification,
        "complete": False,
        "tiles": [],
        "mosaic": mosaic.name,
    }
    write_json(manifest_path, manifest)

    paths = []

    for index, tile_bounds in enumerate(tiles, 1):
        path = directory / f"tile_{index:05d}.tif"
        logger.info("Батиметрия | тайл {}/{}", index, len(tiles))

        if (
            not cfg.bathymetry_force_download
            and path.name in previous
            and raster_valid(path)
        ):
            information = previous[path.name]
            logger.info("Тайл из кеша: {}", path.name)
        else:
            information = fetch_tile(tile_bounds, path, cfg)

        manifest["tiles"].append({
            "file": path.name,
            "bounds": tile_bounds,
            "provider": information["provider"],
            "downloaded_at": information.get("downloaded_at"),
        })

        write_json(manifest_path, manifest)
        paths.append(path)

    resolutions = []

    for path in paths:
        with rasterio.open(path) as src:
            resolutions.append(src.res)

    # При смешанных источниках не увеличиваем искусственно
    # детализацию более грубого источника.
    output_resolution = (
        max(item[0] for item in resolutions),
        max(item[1] for item in resolutions),
    )

    providers = {
        item["provider"] for item in manifest["tiles"]
    }

    if len(providers) > 1:
        logger.warning(
            "Источники смешаны: {}. Мозаика собирается "
            "с наиболее грубым шагом.",
            sorted(providers),
        )

    temporary = directory / "bathymetry.partial.tif"
    temporary.unlink(missing_ok=True)

    try:
        with stage("Сборка мозаики батиметрии"):
            merge(
                [str(path) for path in paths],
                res=output_resolution,
                dtype="float32",
                nodata=float("nan"),
                resampling=Resampling.nearest,
                method="first",
                mem_limit=cfg.bathymetry_merge_memory_mb,
                dst_path=str(temporary),
                dst_kwds={
                    "driver": "GTiff",
                    "compress": "deflate",
                    "tiled": True,
                    "BIGTIFF": "IF_SAFER",
                },
            )

            if not raster_valid(temporary):
                raise ValueError("Некорректная мозаика")

            verify_negative_depths(temporary)
            temporary.replace(mosaic)

    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest["complete"] = True
    manifest["output_resolution_degrees"] = list(output_resolution)
    write_json(manifest_path, manifest)

    logger.success("Готовая батиметрия: {}", mosaic)

    return replace(cfg, bathymetry_path=mosaic)