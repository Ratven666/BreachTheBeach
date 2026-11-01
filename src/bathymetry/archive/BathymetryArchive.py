from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
from loguru import logger
from pyproj import Geod
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.base.BBox import BBox
from src.bathymetry.domain.models import (
    BathymetryGrid,
    GeoLine,
    GeoPoint,
)
from src.bathymetry.exporters.NetCDFBathymetryExporter import (
    NetCDFBathymetryExporter,
)
from src.bathymetry.loaders.BathymetryLoader import (
    BathymetryLoader,
)
from src.bathymetry.loaders.EMODnetBathymetryLoader import (
    EMODnetBathymetryLoader,
)
from src.bathymetry.loaders.GEBCODownloadLoader import (
    GEBCODownloadLoader,
)
from src.bathymetry.loaders.LocalNetCDFBathymetryLoader import (
    LocalNetCDFBathymetryLoader,
)
from src.bathymetry.services.BathymetryService import (
    BathymetryService,
)
from src.coastline.storage.models import (
    CoastlinePointModel,
    CoastlineSourceModel,
    WindFetchModel,
)


_WGS84 = Geod(ellps="WGS84")

Provider = Literal[
    "auto",
    "emodnet",
    "gebco",
]

LoaderFactory = Callable[[BBox], BathymetryLoader]


@dataclass(frozen=True, slots=True)
class BathymetryArchiveConfig:
    """
    Настройки построения локального архива батиметрии.

    max_cells_per_tile ограничивает ожидаемое количество ячеек
    одного запроса. Вдоль береговой линии формируются несколько
    последовательных перекрывающихся тайлов.
    """

    output_dir: Path = Path("data/bathymetry")

    provider: Provider = "auto"

    # Дополнительный запас относительно максимального fetch из БД.
    safety_factor: float = 1.05

    # Дополнительный геодезический буфер вокруг итогового bbox.
    padding_m: float = 1_000.0

    # Ограничение размера одного тайла.
    max_cells_per_tile: int = 4_000_000

    # Используется только для предварительной оценки размера.
    # EMODnet: 1/16 arc minute = 3.75 arc seconds.
    planning_resolution_arcsec: float = 3.75

    # Число береговых точек, повторяющихся в соседних тайлах.
    overlap_points: int = 3

    # Шаг дискретизации геодезического круга.
    circle_step_deg: float = 2.0

    retries: int = 3
    retry_delay_s: float = 5.0

    # Перезаписывать уже существующие NetCDF.
    force: bool = False

    # Ограничение максимального fetch_length_m.
    # None → брать из БД как есть.
    # float > 0 → использовать это значение вместо значения из БД
    #             для всех точек; реальный fetch из БД, превышающий
    #             этот порог, будет обрезан до него.
    #             Значения меньше порога остаются без изменений.
    max_fetch_override_m: float | None = None

    def __post_init__(self) -> None:
        if self.safety_factor < 1.0:
            raise ValueError(
                "safety_factor must be >= 1.0"
            )

        if self.padding_m < 0:
            raise ValueError(
                "padding_m must be >= 0"
            )

        if self.max_cells_per_tile < 4:
            raise ValueError(
                "max_cells_per_tile must be >= 4"
            )

        if self.planning_resolution_arcsec <= 0:
            raise ValueError(
                "planning_resolution_arcsec must be > 0"
            )

        if self.overlap_points < 0:
            raise ValueError(
                "overlap_points must be >= 0"
            )

        if not (0 < self.circle_step_deg <= 30):
            raise ValueError(
                "circle_step_deg must be in (0, 30]"
            )

        if self.retries < 1:
            raise ValueError(
                "retries must be >= 1"
            )

        if (
            self.max_fetch_override_m is not None
            and not (
                math.isfinite(self.max_fetch_override_m)
                and self.max_fetch_override_m > 0
            )
        ):
            raise ValueError(
                "max_fetch_override_m must be > 0 or None"
            )


@dataclass(frozen=True, slots=True)
class PointCoverage:
    """
    Требуемое покрытие для одной береговой точки.
    """

    point_id: int
    seq: int

    lon: float
    lat: float

    max_fetch_m: float

    # Геодезический bbox круга радиуса:
    # max_fetch_m * safety_factor.
    required_bbox: BBox


@dataclass(frozen=True, slots=True)
class BathymetryTilePlan:
    """
    Описание одного запрашиваемого тайла.
    """

    tile_id: str

    point_ids: tuple[int, ...]

    seq_min: int
    seq_max: int

    max_fetch_m: float

    # Минимальный bbox, необходимый для покрытия fetch.
    required_bbox: BBox

    # Реально запрашиваемый bbox с padding_m.
    request_bbox: BBox

    estimated_cells: int


@dataclass(frozen=True, slots=True)
class StoredBathymetryProfile:
    """
    Батиметрический профиль в интерфейсе,
    совместимом с WaveClimateService.
    """

    direction: int
    distances_m: np.ndarray

    # Положительные значения означают глубину.
    # Суша и нулевые высоты преобразуются в NaN.
    depths_m: np.ndarray


class BathymetryCoveragePlanner:
    """
    Строит небольшое количество перекрывающихся bbox вдоль
    последовательности береговых точек.

    Для каждой точки из coastline_points выбирается максимальный
    fetch_length_m из wind_fetches.

    Вокруг точки строится геодезический круг радиусом:

        max_fetch_m * safety_factor

    Объединение кругов последовательных точек формирует один тайл.
    Добавление точек прекращается, когда расчётное число ячеек
    превышает max_cells_per_tile.
    """

    def __init__(
        self,
        session: Session,
        config: BathymetryArchiveConfig,
    ) -> None:
        self._session = session
        self._config = config

    def resolve_point_source_id(
        self,
        point_source_id: int | None,
    ) -> int:
        """
        Если id не передан, выбирает последний источник,
        для которого существуют рассчитанные wind_fetches.
        """
        if point_source_id is not None:
            source = self._session.get(
                CoastlineSourceModel,
                point_source_id,
            )

            if source is None:
                raise ValueError(
                    f"CoastlineSource id={point_source_id} "
                    "not found"
                )

            return int(point_source_id)

        value = self._session.execute(
            select(
                CoastlinePointModel.source_id
            )
            .join(
                WindFetchModel,
                WindFetchModel.point_id
                == CoastlinePointModel.id,
            )
            .group_by(
                CoastlinePointModel.source_id
            )
            .order_by(
                CoastlinePointModel.source_id.desc()
            )
            .limit(1)
        ).scalar_one_or_none()

        if value is None:
            raise ValueError(
                "No coastline source with wind fetches found. "
                "Run database pipeline steps 1-3 first."
            )

        return int(value)

    def load_point_coverages(
        self,
        point_source_id: int,
    ) -> list[PointCoverage]:
        """
        Загружает точки и максимальные радиусы разгона.

        Наличие fetch проверяется для каждой точки. Частично
        рассчитанный набор считается ошибочным, поскольку в этом
        случае нельзя гарантировать покрытие батиметрией.

        Если задан max_fetch_override_m, значение fetch из БД
        ограничивается этим порогом сверху; меньшие значения
        остаются без изменений.
        """
        rows = self._session.execute(
            select(
                CoastlinePointModel.id,
                CoastlinePointModel.seq,
                CoastlinePointModel.lon,
                CoastlinePointModel.lat,
                func.max(
                    WindFetchModel.fetch_length_m
                ),
            )
            .outerjoin(
                WindFetchModel,
                WindFetchModel.point_id
                == CoastlinePointModel.id,
            )
            .where(
                CoastlinePointModel.source_id
                == point_source_id
            )
            .group_by(
                CoastlinePointModel.id,
                CoastlinePointModel.seq,
                CoastlinePointModel.lon,
                CoastlinePointModel.lat,
            )
            .order_by(
                CoastlinePointModel.seq,
                CoastlinePointModel.id,
            )
        ).all()

        if not rows:
            raise ValueError(
                f"CoastlineSource id={point_source_id} "
                "has no points"
            )

        missing = [
            int(row[0])
            for row in rows
            if row[4] is None
        ]

        if missing:
            raise ValueError(
                "Wind fetches are missing for coastline points: "
                f"{missing[:20]}. "
                "Run db_pypeline/3_1_wind_fetch_to_db.py first."
            )

        result: list[PointCoverage] = []

        override = self._config.max_fetch_override_m

        for (
            point_id,
            seq,
            lon,
            lat,
            max_fetch,
        ) in rows:
            db_fetch_m = float(max_fetch)

            if (
                not math.isfinite(db_fetch_m)
                or db_fetch_m <= 0
            ):
                raise ValueError(
                    "Invalid maximum fetch for "
                    f"point_id={point_id}: {max_fetch}"
                )

            # Применяем override: он ограничивает радиус сверху,
            # но никогда не расширяет его за пределы данных БД.
            if override is not None and db_fetch_m > override:
                radius_m = override
            else:
                radius_m = db_fetch_m

            protected_radius = (
                radius_m
                * self._config.safety_factor
            )

            bbox = self._geodesic_circle_bbox(
                lon=float(lon),
                lat=float(lat),
                radius_m=protected_radius,
            )

            result.append(
                PointCoverage(
                    point_id=int(point_id),
                    seq=int(seq),
                    lon=float(lon),
                    lat=float(lat),
                    max_fetch_m=radius_m,
                    required_bbox=bbox,
                )
            )

        return result

    def plan(
        self,
        point_source_id: int | None = None,
    ) -> tuple[
        int,
        list[PointCoverage],
        list[BathymetryTilePlan],
    ]:
        source_id = self.resolve_point_source_id(
            point_source_id
        )

        points = self.load_point_coverages(
            source_id
        )

        chunks = self._split_with_overlap(
            points
        )

        tiles: list[BathymetryTilePlan] = []

        for number, chunk in enumerate(
            chunks,
            start=1,
        ):
            required_bbox = _union_bbox(
                point.required_bbox
                for point in chunk
            )

            request_bbox = _expand_bbox(
                required_bbox,
                self._config.padding_m,
            )

            estimated_cells = _estimate_cells(
                request_bbox,
                self._config.planning_resolution_arcsec,
            )

            tiles.append(
                BathymetryTilePlan(
                    tile_id=f"tile_{number:03d}",
                    point_ids=tuple(
                        point.point_id
                        for point in chunk
                    ),
                    seq_min=min(
                        point.seq
                        for point in chunk
                    ),
                    seq_max=max(
                        point.seq
                        for point in chunk
                    ),
                    max_fetch_m=max(
                        point.max_fetch_m
                        for point in chunk
                    ),
                    required_bbox=required_bbox,
                    request_bbox=request_bbox,
                    estimated_cells=estimated_cells,
                )
            )

        return source_id, points, tiles

    def _split_with_overlap(
        self,
        points: list[PointCoverage],
    ) -> list[list[PointCoverage]]:
        chunks: list[list[PointCoverage]] = []

        start = 0

        while start < len(points):
            chunk: list[PointCoverage] = []
            index = start

            while index < len(points):
                candidate = [
                    *chunk,
                    points[index],
                ]

                candidate_bbox = _union_bbox(
                    point.required_bbox
                    for point in candidate
                )

                request_bbox = _expand_bbox(
                    candidate_bbox,
                    self._config.padding_m,
                )

                estimated_cells = _estimate_cells(
                    request_bbox,
                    self._config
                    .planning_resolution_arcsec,
                )

                if (
                    chunk
                    and estimated_cells
                    > self._config.max_cells_per_tile
                ):
                    break

                if (
                    not chunk
                    and estimated_cells
                    > self._config.max_cells_per_tile
                ):
                    point = points[index]

                    raise ValueError(
                        "A single point "
                        f"(point_id={point.point_id}) needs "
                        f"approximately {estimated_cells:,} cells, "
                        "which is above max_cells_per_tile="
                        f"{self._config.max_cells_per_tile:,}. "
                        "Increase the limit or use a coarser "
                        "provider."
                    )

                chunk.append(
                    points[index]
                )

                index += 1

            chunks.append(chunk)

            if index >= len(points):
                break

            overlap = min(
                self._config.overlap_points,
                max(
                    0,
                    len(chunk) - 1,
                ),
            )

            start = index - overlap

        return chunks

    def _geodesic_circle_bbox(
        self,
        lon: float,
        lat: float,
        radius_m: float,
    ) -> BBox:
        """
        Строит bbox геодезического круга на WGS84.

        Это надёжнее приближённого перевода метров в градусы,
        особенно для долгот на высоких широтах.
        """
        bearings = np.arange(
            0.0,
            360.0,
            self._config.circle_step_deg,
        )

        circle_lons, circle_lats, _ = _WGS84.fwd(
            np.full(
                bearings.shape,
                lon,
            ),
            np.full(
                bearings.shape,
                lat,
            ),
            bearings,
            np.full(
                bearings.shape,
                radius_m,
            ),
        )

        all_lons = np.append(
            np.asarray(
                circle_lons,
                dtype=float,
            ),
            lon,
        )

        all_lats = np.append(
            np.asarray(
                circle_lats,
                dtype=float,
            ),
            lat,
        )

        if float(np.ptp(all_lons)) > 180.0:
            raise ValueError(
                "Bathymetry tiling across the "
                "antimeridian is not supported"
            )

        return _safe_bbox(
            south=float(all_lats.min()),
            west=float(all_lons.min()),
            north=float(all_lats.max()),
            east=float(all_lons.max()),
        )


class BathymetryArchiveBuilder:
    """
    Скачивает запланированные тайлы и создаёт manifest.json.

    Каждый сохранённый NetCDF проверяется на покрытие required_bbox.
    Неполный ответ провайдера не принимается.
    """

    def __init__(
        self,
        session: Session,
        config: BathymetryArchiveConfig,
        loader_factory: LoaderFactory | None = None,
    ) -> None:
        self._session = session
        self._config = config

        self._loader_factory = (
            loader_factory
            or self._default_loader
        )

        self._log = logger.bind(
            cls=self.__class__.__name__
        )

    def build(
        self,
        point_source_id: int | None = None,
    ) -> Path:
        planner = BathymetryCoveragePlanner(
            session=self._session,
            config=self._config,
        )

        (
            source_id,
            points,
            plans,
        ) = planner.plan(
            point_source_id
        )

        output_dir = (
            self._config.output_dir.resolve()
        )

        tile_dir = output_dir / "tiles"

        tile_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tile_records: list[dict] = []

        point_to_tiles: dict[
            int,
            list[str],
        ] = defaultdict(list)

        for plan in plans:
            path = (
                tile_dir
                / f"{plan.tile_id}.nc"
            )

            grid = self._download_or_open(
                plan=plan,
                path=path,
            )

            self._assert_grid_covers(
                required=plan.required_bbox,
                grid=grid,
                tile_id=plan.tile_id,
            )

            for point_id in plan.point_ids:
                point_to_tiles[
                    point_id
                ].append(
                    plan.tile_id
                )

            tile_records.append(
                {
                    "tile_id": plan.tile_id,
                    "file": path.relative_to(
                        output_dir
                    ).as_posix(),
                    "point_ids": list(
                        plan.point_ids
                    ),
                    "seq_min": plan.seq_min,
                    "seq_max": plan.seq_max,
                    "max_fetch_m": (
                        plan.max_fetch_m
                    ),
                    "required_bbox": _bbox_dict(
                        plan.required_bbox
                    ),
                    "request_bbox": _bbox_dict(
                        plan.request_bbox
                    ),
                    "grid_bbox": _bbox_dict(
                        BBox(
                            south=grid.south,
                            west=grid.west,
                            north=grid.north,
                            east=grid.east,
                        )
                    ),
                    "estimated_cells": (
                        plan.estimated_cells
                    ),
                    "shape": list(
                        grid.shape
                    ),
                    "source": grid.source,
                }
            )

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(
                UTC
            ).isoformat(),
            "point_source_id": source_id,
            "config": {
                **asdict(
                    self._config
                ),
                "output_dir": str(
                    self._config.output_dir
                ),
            },
            "points": {
                str(point.point_id): {
                    "seq": point.seq,
                    "lon": point.lon,
                    "lat": point.lat,
                    "max_fetch_m": (
                        point.max_fetch_m
                    ),
                    "tile_ids": point_to_tiles[
                        point.point_id
                    ],
                }
                for point in points
            },
            "tiles": tile_records,
        }

        manifest_path = (
            output_dir
            / "manifest.json"
        )

        temporary_path = (
            output_dir
            / "manifest.json.tmp"
        )

        temporary_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            manifest_path,
        )

        self._log.success(
            "Bathymetry archive ready: "
            f"{len(points)} points, "
            f"{len(plans)} tiles, "
            f"manifest={manifest_path}"
        )

        return manifest_path

    def _download_or_open(
        self,
        plan: BathymetryTilePlan,
        path: Path,
    ) -> BathymetryGrid:
        if (
            path.exists()
            and not self._config.force
        ):
            self._log.info(
                f"Reusing {path}"
            )

            return _load_full_local_grid(
                path
            )

        last_error: Exception | None = None

        for attempt in range(
            1,
            self._config.retries + 1,
        ):
            try:
                loader = self._loader_factory(
                    plan.request_bbox
                )

                service = BathymetryService(
                    loader=loader
                )

                grid = service.fetch(
                    plan.request_bbox,
                    force_reload=True,
                )

                self._assert_grid_covers(
                    required=plan.required_bbox,
                    grid=grid,
                    tile_id=plan.tile_id,
                )

                temporary_path = (
                    path.with_suffix(
                        ".part.nc"
                    )
                )

                NetCDFBathymetryExporter().export(
                    grid,
                    temporary_path,
                )

                os.replace(
                    temporary_path,
                    path,
                )

                return grid

            except Exception as exc:
                last_error = exc

                path.with_suffix(
                    ".part.nc"
                ).unlink(
                    missing_ok=True
                )

                if (
                    attempt
                    < self._config.retries
                ):
                    delay = (
                        self._config.retry_delay_s
                        * attempt
                    )

                    self._log.warning(
                        f"{plan.tile_id}: attempt "
                        f"{attempt} failed: {exc}; "
                        f"retry in {delay:.1f}s"
                    )

                    time.sleep(delay)

        assert last_error is not None

        raise RuntimeError(
            f"Failed to download {plan.tile_id} "
            f"after {self._config.retries} attempts"
        ) from last_error

    def _default_loader(
        self,
        bbox: BBox,
    ) -> BathymetryLoader:
        provider = self._config.provider

        if (
            provider == "emodnet"
            or (
                provider == "auto"
                and _within_emodnet_coverage(
                    bbox
                )
            )
        ):
            return EMODnetBathymetryLoader(
                timeout=300
            )

        return GEBCODownloadLoader(
            timeout=300
        )

    @staticmethod
    def _assert_grid_covers(
        required: BBox,
        grid: BathymetryGrid,
        tile_id: str,
    ) -> None:
        """
        Проверяет именно required_bbox, а не request_bbox.

        Запрошенный padding может частично выходить за физическое
        покрытие провайдера, но все требуемые fetch-радиусы обязаны
        оставаться внутри полученного грида.
        """
        eps = 1e-10

        covered = (
            grid.south
            <= required.south + eps
            and grid.west
            <= required.west + eps
            and grid.north
            >= required.north - eps
            and grid.east
            >= required.east - eps
        )

        if covered:
            return

        grid_bbox = BBox(
            south=grid.south,
            west=grid.west,
            north=grid.north,
            east=grid.east,
        )

        raise ValueError(
            f"{tile_id}: downloaded grid does not cover "
            "the required fetch area. "
            f"required={required}, grid={grid_bbox}"
        )


class BathymetryArchive:
    """
    Локальный read-only архив батиметрии.

    Объект совместим с WaveClimateBatchProcessor, поскольку
    предоставляет метод:

        for_point(lon, lat)

    Возвращаемый объект предоставляет:

        get_profile(direction)
        profile.depths_m
    """

    def __init__(
        self,
        manifest_path: str | Path,
        coordinate_tolerance_m: float = 5.0,
    ) -> None:
        self.manifest_path = Path(
            manifest_path
        ).resolve()

        self.root_dir = (
            self.manifest_path.parent
        )

        self._data = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            self._data.get(
                "schema_version"
            )
            != 1
        ):
            raise ValueError(
                "Unsupported bathymetry "
                "manifest schema"
            )

        self._points: dict[
            str,
            dict,
        ] = self._data["points"]

        self._tiles = {
            tile["tile_id"]: tile
            for tile in self._data["tiles"]
        }

        self._services: dict[
            str,
            BathymetryService,
        ] = {}

        self._coordinate_tolerance_m = (
            coordinate_tolerance_m
        )

        self._coordinate_index: dict[
            tuple[float, float],
            list[int],
        ] = defaultdict(list)

        for (
            point_id,
            value,
        ) in self._points.items():
            coordinate = (
                round(
                    float(value["lon"]),
                    7,
                ),
                round(
                    float(value["lat"]),
                    7,
                ),
            )

            self._coordinate_index[
                coordinate
            ].append(
                int(point_id)
            )

        self._validate_files()

    @property
    def point_source_id(self) -> int:
        return int(
            self._data[
                "point_source_id"
            ]
        )

    @property
    def point_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(value)
                for value in self._points
            )
        )

    def point_metadata(
        self,
        point_id: int,
    ) -> dict:
        point = self._points.get(
            str(int(point_id))
        )

        if point is None:
            raise KeyError(
                f"point_id={point_id} is absent "
                "from bathymetry archive"
            )

        return dict(point)

    def for_point_id(
        self,
        point_id: int,
        *,
        radius_m: float | None = None,
        n_steps: int = 200,
    ) -> ArchivedPointBathymetryService:
        point = self._points.get(
            str(int(point_id))
        )

        if point is None:
            raise KeyError(
                f"point_id={point_id} is absent "
                "from bathymetry archive"
            )

        max_fetch_m = float(
            point["max_fetch_m"]
        )

        radius = (
            max_fetch_m
            if radius_m is None
            else float(radius_m)
        )

        if radius <= 0:
            raise ValueError(
                "radius_m must be > 0"
            )

        if radius > max_fetch_m + 1e-6:
            raise ValueError(
                f"radius_m={radius} exceeds archived "
                f"max_fetch_m={max_fetch_m} "
                f"for point_id={point_id}"
            )

        tile_id = str(
            point["tile_ids"][0]
        )

        return ArchivedPointBathymetryService(
            service=self._service_for_tile(
                tile_id
            ),
            point_id=int(point_id),
            lon=float(point["lon"]),
            lat=float(point["lat"]),
            radius_m=radius,
            n_steps=n_steps,
        )

    def for_point(
        self,
        lon: float,
        lat: float,
        *,
        radius_m: float | None = None,
        n_steps: int = 200,
    ) -> ArchivedPointBathymetryService:
        """
        Находит точку архива по координатам.

        Этот метод автоматически вызывается существующим
        WaveClimateBatchProcessor.
        """
        coordinate = (
            round(
                float(lon),
                7,
            ),
            round(
                float(lat),
                7,
            ),
        )

        candidates = self._coordinate_index.get(
            coordinate,
            [],
        )

        if not candidates:
            candidates = [
                int(point_id)
                for (
                    point_id,
                    point,
                ) in self._points.items()
                if _distance_m(
                    lon,
                    lat,
                    float(point["lon"]),
                    float(point["lat"]),
                )
                <= self._coordinate_tolerance_m
            ]

        if len(candidates) != 1:
            raise KeyError(
                "Expected one archived point near "
                f"({lon}, {lat}), "
                f"found {len(candidates)}"
            )

        return self.for_point_id(
            candidates[0],
            radius_m=radius_m,
            n_steps=n_steps,
        )

    def _service_for_tile(
        self,
        tile_id: str,
    ) -> BathymetryService:
        if tile_id in self._services:
            return self._services[
                tile_id
            ]

        tile = self._tiles[
            tile_id
        ]

        path = (
            self.root_dir
            / tile["file"]
        )

        grid_bbox = _bbox_from_dict(
            tile["grid_bbox"]
        )

        loader = (
            LocalNetCDFBathymetryLoader(
                path,
                source_name=(
                    f"archive:{tile_id}"
                ),
            )
        )

        service = BathymetryService(
            loader=loader
        )

        service.fetch(
            grid_bbox
        )

        self._services[
            tile_id
        ] = service

        return service

    def _validate_files(self) -> None:
        missing = [
            str(
                self.root_dir
                / tile["file"]
            )
            for tile in self._tiles.values()
            if not (
                self.root_dir
                / tile["file"]
            ).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing bathymetry tiles: "
                f"{missing[:10]}"
            )


class ArchivedPointBathymetryService:
    """
    Батиметрический сервис, привязанный к конкретной точке.

    По умолчанию длина профиля равна максимальному fetch данной
    точки. Профиль никогда не может быть запрошен за пределами
    радиуса, сохранённого в manifest.
    """

    def __init__(
        self,
        *,
        service: BathymetryService,
        point_id: int,
        lon: float,
        lat: float,
        radius_m: float,
        n_steps: int,
    ) -> None:
        if n_steps < 2:
            raise ValueError(
                "n_steps must be >= 2"
            )

        self.point_id = point_id

        self.lon = lon
        self.lat = lat

        self.radius_m = radius_m
        self.n_steps = n_steps

        self._service = service

        self._cache: dict[
            int,
            StoredBathymetryProfile,
        ] = {}

    def get_profile(
        self,
        direction: int,
    ) -> StoredBathymetryProfile:
        direction = (
            int(direction)
            % 360
        )

        if direction in self._cache:
            return self._cache[
                direction
            ]

        (
            end_lon,
            end_lat,
            _,
        ) = _WGS84.fwd(
            self.lon,
            self.lat,
            float(direction),
            self.radius_m,
        )

        line = GeoLine(
            start=GeoPoint(
                lat=self.lat,
                lon=self.lon,
            ),
            end=GeoPoint(
                lat=float(end_lat),
                lon=float(end_lon),
            ),
        )

        profile = (
            self._service.build_profile(
                line,
                n_points=self.n_steps,
            )
        )

        elevation = np.asarray(
            profile.depths,
            dtype=np.float64,
        )

        # Батиметрия хранится как elevation:
        # отрицательное значение — дно.
        # WaveClimateService ожидает положительную глубину.
        depths = np.where(
            elevation < 0.0,
            -elevation,
            np.nan,
        )

        result = StoredBathymetryProfile(
            direction=direction,
            distances_m=np.asarray(
                profile.distances,
                dtype=np.float64,
            ),
            depths_m=depths,
        )

        self._cache[
            direction
        ] = result

        return result


def _within_emodnet_coverage(
    bbox: BBox,
) -> bool:
    """
    Те же пределы, которые использует существующая
    BathymetryLoaderFactory проекта.
    """
    return (
        15.0 <= bbox.south
        and -36.0 <= bbox.west
        and bbox.north <= 90.0
        and bbox.east <= 43.0
    )


def _distance_m(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    return float(
        _WGS84.inv(
            lon1,
            lat1,
            lon2,
            lat2,
        )[2]
    )


def _safe_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
) -> BBox:
    eps = 1e-9

    if north <= south:
        north = south + eps

    if east <= west:
        east = west + eps

    return BBox(
        south=max(
            -90.0,
            south,
        ),
        west=max(
            -180.0,
            west,
        ),
        north=min(
            90.0,
            north,
        ),
        east=min(
            180.0,
            east,
        ),
    )


def _union_bbox(
    boxes: Iterable[BBox],
) -> BBox:
    items = list(boxes)

    if not items:
        raise ValueError(
            "Cannot build bbox from "
            "an empty collection"
        )

    return BBox(
        south=min(
            bbox.south
            for bbox in items
        ),
        west=min(
            bbox.west
            for bbox in items
        ),
        north=max(
            bbox.north
            for bbox in items
        ),
        east=max(
            bbox.east
            for bbox in items
        ),
    )


def _expand_bbox(
    bbox: BBox,
    padding_m: float,
) -> BBox:
    """
    Расширяет bbox геодезически, а не через постоянный
    коэффициент метров на градус.
    """
    if padding_m == 0:
        return bbox

    corners = [
        (
            bbox.west,
            bbox.south,
        ),
        (
            bbox.west,
            bbox.north,
        ),
        (
            bbox.east,
            bbox.south,
        ),
        (
            bbox.east,
            bbox.north,
        ),
    ]

    lons = [
        point[0]
        for point in corners
    ]

    lats = [
        point[1]
        for point in corners
    ]

    for lon, lat in corners:
        for bearing in (
            0.0,
            90.0,
            180.0,
            270.0,
        ):
            (
                lon2,
                lat2,
                _,
            ) = _WGS84.fwd(
                lon,
                lat,
                bearing,
                padding_m,
            )

            lons.append(
                float(lon2)
            )

            lats.append(
                float(lat2)
            )

    if max(lons) - min(lons) > 180.0:
        raise ValueError(
            "Expanded bbox crosses "
            "the antimeridian"
        )

    return _safe_bbox(
        south=min(lats),
        west=min(lons),
        north=max(lats),
        east=max(lons),
    )


def _estimate_cells(
    bbox: BBox,
    resolution_arcsec: float,
) -> int:
    nx = (
        math.ceil(
            (
                bbox.east
                - bbox.west
            )
            * 3600.0
            / resolution_arcsec
        )
        + 1
    )

    ny = (
        math.ceil(
            (
                bbox.north
                - bbox.south
            )
            * 3600.0
            / resolution_arcsec
        )
        + 1
    )

    return int(
        nx * ny
    )


def _bbox_dict(
    bbox: BBox,
) -> dict[str, float]:
    return {
        "south": bbox.south,
        "west": bbox.west,
        "north": bbox.north,
        "east": bbox.east,
    }


def _bbox_from_dict(
    value: dict,
) -> BBox:
    return BBox(
        south=float(
            value["south"]
        ),
        west=float(
            value["west"]
        ),
        north=float(
            value["north"]
        ),
        east=float(
            value["east"]
        ),
    )


def _load_full_local_grid(
    path: Path,
) -> BathymetryGrid:
    """
    Загружает существующий локальный тайл целиком.

    Используется при повторном запуске без BATHY_FORCE.
    """
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        lats = np.asarray(
            dataset["lat"].values,
            dtype=np.float64,
        )

        lons = np.asarray(
            dataset["lon"].values,
            dtype=np.float64,
        )

    bbox = BBox(
        south=float(
            lats.min()
        ),
        west=float(
            lons.min()
        ),
        north=float(
            lats.max()
        ),
        east=float(
            lons.max()
        ),
    )

    loader = LocalNetCDFBathymetryLoader(
        path
    )

    return loader.load(
        bbox
    )
