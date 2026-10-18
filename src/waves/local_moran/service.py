from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .calc import (
    PreparedGeometry,
    apply_bh_fdr,
    assign_points_to_coastline,
    compute_local_moran_for_metric,
    filter_snap_ok_rows,
    resolve_alongshore_metric,
)
from .repository import (
    LocalMoranRepository,
    sha256_of_file,
)


_ALLOWED_WEIGHT_MODES = {
    "binary",
    "inverse_distance",
}


@dataclass(frozen=True, slots=True)
class LocalMoranStats:
    """
    Итоговая статистика одного запуска Local Moran.

    Attributes
    ----------
    run_id
        Идентификатор запуска в таблице local_moran_runs.
    n_points_total
        Общее количество точек, прочитанных из исходной базы.
    n_points_snap_ok
        Количество точек, прошедших проверку расстояния до берега.
    n_points_snap_bad
        Количество точек, удалённых из расчёта по snap-расстоянию.
    n_results
        Количество записей, добавленных в local_moran_results.
        Обычно равно n_points_snap_ok * 2, так как рассчитываются
        нормальная и вдольбереговая метрики.
    output_db
        Путь к дополненной базе wave_activity_indices.db.
    """

    run_id: int
    n_points_total: int
    n_points_snap_ok: int
    n_points_snap_bad: int
    n_results: int
    output_db: Path

    @property
    def n_points(self) -> int:
        """
        Совместимость с более ранней версией вызывающего скрипта.
        """
        return self.n_points_total


class LocalMoranService:
    """
    Сервис расчёта локального индекса Морана вдоль береговой линии.

    Сервис выполняет следующие операции:

    1. Читает координаты и средние показатели из
       wave_activity_indices.db.
    2. Назначает каждой точке ближайший сегмент переданной линии.
    3. Вычисляет линейную координату точки вдоль сегмента.
    4. Проверяет расстояние точки до береговой линии.
    5. Рассчитывает Local Moran отдельно для:
       - normal_mean_wm;
       - alongshore_abs_mean_wm или alongshore_net_mean_wm.
    6. Выполняет перестановочный тест.
    7. Выполняет FDR-коррекцию Benjamini–Hochberg.
    8. Дополняет исходную SQLite-базу результатами.

    Parameters
    ----------
    db_path
        Путь к wave_activity_indices.db.
    coastline_geojson
        Путь к GeoJSON с береговой линией.
    radius_m
        Максимальное вдольбереговое расстояние до соседей.
    window_width_m
        Полная ширина скользящего окна.
    alongshore_mode
        ``"abs"`` для alongshore_abs_mean_wm или
        ``"net"`` для alongshore_net_mean_wm.
    metric_crs
        Проекционная CRS. Если None, calc.py оценивает UTM автоматически.
    max_snap_distance_m
        Максимально допустимое расстояние от точки до линии.
        Если None, все точки считаются прошедшими snap-проверку.
    strict_snap
        Если True, наличие хотя бы одной слишком удалённой точки
        прерывает расчёт. Если False, такие точки сохраняются в таблице
        порядка, но исключаются из статистического анализа.
    weight_mode
        ``"binary"`` или ``"inverse_distance"``.
    permutations
        Число условных случайных перестановок.
    seed
        Базовое значение генератора случайных чисел.
    alpha_raw
        Уровень значимости для исходного pseudo p-value и поля cluster.
    alpha_fdr
        Уровень значимости для BH-adjusted q-value и поля cluster_fdr.
    """

    def __init__(
        self,
        db_path: str | Path,
        coastline_geojson: str | Path,
        *,
        radius_m: float,
        window_width_m: float,
        alongshore_mode: str = "abs",
        metric_crs: str | None = None,
        max_snap_distance_m: float | None = None,
        strict_snap: bool = False,
        weight_mode: str = "binary",
        permutations: int = 999,
        seed: int = 42,
        alpha_raw: float = 0.05,
        alpha_fdr: float = 0.05,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.coastline_geojson = (
            Path(coastline_geojson).expanduser().resolve()
        )

        self.radius_m = float(radius_m)
        self.window_width_m = float(window_width_m)

        self.alongshore_mode = (
            str(alongshore_mode).strip().lower()
        )
        self.alongshore_metric = resolve_alongshore_metric(
            self.alongshore_mode
        )

        self.metric_crs = (
            None
            if metric_crs is None
            else str(metric_crs).strip()
        )

        self.max_snap_distance_m = (
            None
            if max_snap_distance_m is None
            else float(max_snap_distance_m)
        )

        self.strict_snap = bool(strict_snap)
        self.weight_mode = str(weight_mode).strip().lower()
        self.permutations = int(permutations)
        self.seed = int(seed)
        self.alpha_raw = float(alpha_raw)
        self.alpha_fdr = float(alpha_fdr)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Проверить параметры до чтения файлов и запуска вычислений.
        """
        if self.radius_m <= 0.0:
            raise ValueError(
                f"radius_m must be > 0, got {self.radius_m}"
            )

        if self.window_width_m <= 0.0:
            raise ValueError(
                "window_width_m must be > 0, "
                f"got {self.window_width_m}"
            )

        if (
            self.window_width_m + 1.0e-9
            < 2.0 * self.radius_m
        ):
            raise ValueError(
                "window_width_m must be at least "
                "2 * radius_m. "
                f"Got radius_m={self.radius_m}, "
                f"window_width_m={self.window_width_m}"
            )

        if (
            self.max_snap_distance_m is not None
            and self.max_snap_distance_m <= 0.0
        ):
            raise ValueError(
                "max_snap_distance_m must be > 0 "
                "or None, got "
                f"{self.max_snap_distance_m}"
            )

        if self.weight_mode not in _ALLOWED_WEIGHT_MODES:
            raise ValueError(
                f"Unsupported weight_mode={self.weight_mode!r}. "
                f"Expected one of "
                f"{sorted(_ALLOWED_WEIGHT_MODES)}"
            )

        if self.permutations < 99:
            raise ValueError(
                "permutations must be >= 99, "
                f"got {self.permutations}"
            )

        if self.seed < 0:
            raise ValueError(
                f"seed must be >= 0, got {self.seed}"
            )

        if not 0.0 < self.alpha_raw < 1.0:
            raise ValueError(
                "alpha_raw must be between 0 and 1, "
                f"got {self.alpha_raw}"
            )

        if not 0.0 < self.alpha_fdr < 1.0:
            raise ValueError(
                "alpha_fdr must be between 0 and 1, "
                f"got {self.alpha_fdr}"
            )

        if (
            self.metric_crs is not None
            and not self.metric_crs
        ):
            raise ValueError(
                "metric_crs must be a valid CRS string or None"
            )

    def _validate_input_files(self) -> None:
        """
        Проверить существование входной базы и береговой линии.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Index database not found: {self.db_path}"
            )

        if not self.db_path.is_file():
            raise ValueError(
                f"Index database path is not a file: "
                f"{self.db_path}"
            )

        if not self.coastline_geojson.exists():
            raise FileNotFoundError(
                "Coastline GeoJSON not found: "
                f"{self.coastline_geojson}"
            )

        if not self.coastline_geojson.is_file():
            raise ValueError(
                f"Coastline path is not a file: "
                f"{self.coastline_geojson}"
            )

    def _log_parameters(self) -> None:
        logger.info("Index database: {}", self.db_path)
        logger.info(
            "Coastline GeoJSON: {}",
            self.coastline_geojson,
        )
        logger.info(
            "Parameters: radius_m={}, window_width_m={}, "
            "alongshore_mode={}, alongshore_metric={}, "
            "metric_crs={}, max_snap_distance_m={}, "
            "strict_snap={}, weight_mode={}, permutations={}, "
            "seed={}, alpha_raw={}, alpha_fdr={}",
            self.radius_m,
            self.window_width_m,
            self.alongshore_mode,
            self.alongshore_metric,
            self.metric_crs,
            self.max_snap_distance_m,
            self.strict_snap,
            self.weight_mode,
            self.permutations,
            self.seed,
            self.alpha_raw,
            self.alpha_fdr,
        )

    @staticmethod
    def _validate_source_rows(
        source_rows: list[dict],
    ) -> None:
        """
        Проверить данные, прочитанные из points и point_trends.
        """
        if not source_rows:
            raise ValueError(
                "No source points found in index database"
            )

        required_columns = {
            "point_id",
            "lon",
            "lat",
            "normal_mean_wm",
            "alongshore_mean_wm",
        }

        missing_by_row: list[tuple[int, list[str]]] = []

        for row_number, row in enumerate(source_rows):
            missing = sorted(required_columns - set(row))

            if missing:
                missing_by_row.append(
                    (row_number, missing)
                )

            if len(missing_by_row) >= 10:
                break

        if missing_by_row:
            raise ValueError(
                "Source rows do not contain all required "
                f"columns. Sample={missing_by_row}"
            )

        point_ids = [
            int(row["point_id"])
            for row in source_rows
        ]

        if len(point_ids) != len(set(point_ids)):
            duplicate_counts = Counter(point_ids)

            duplicate_ids = [
                point_id
                for point_id, count in duplicate_counts.items()
                if count > 1
            ]

            raise ValueError(
                "Duplicate point_id values found in source "
                f"data. Sample={duplicate_ids[:10]}"
            )

        invalid_coordinates: list[int] = []

        for row in source_rows:
            point_id = int(row["point_id"])

            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (TypeError, ValueError):
                invalid_coordinates.append(point_id)
                continue

            if not (
                -180.0 <= lon <= 180.0
                and -90.0 <= lat <= 90.0
            ):
                invalid_coordinates.append(point_id)

        if invalid_coordinates:
            raise ValueError(
                "Invalid geographic coordinates found. "
                f"Sample point_ids={invalid_coordinates[:10]}"
            )

    @staticmethod
    def _validate_prepared_geometry(
        prepared: PreparedGeometry,
    ) -> None:
        """
        Проверить внутреннюю согласованность результата привязки.
        """
        if prepared.n_total != len(prepared.point_rows):
            raise RuntimeError(
                "PreparedGeometry is inconsistent: "
                f"n_total={prepared.n_total}, "
                f"len(point_rows)={len(prepared.point_rows)}"
            )

        if (
            prepared.n_snap_ok + prepared.n_snap_bad
            != prepared.n_total
        ):
            raise RuntimeError(
                "PreparedGeometry snap counters are "
                "inconsistent: "
                f"n_total={prepared.n_total}, "
                f"n_snap_ok={prepared.n_snap_ok}, "
                f"n_snap_bad={prepared.n_snap_bad}"
            )

        point_ids = [
            int(row["point_id"])
            for row in prepared.point_rows
        ]

        if len(point_ids) != len(set(point_ids)):
            raise RuntimeError(
                "PreparedGeometry contains duplicate point_id "
                "values"
            )

    def _compute_normal(
        self,
        analysis_rows: list[dict],
    ) -> list[dict]:
        logger.info(
            "Computing Local Moran for normal_mean_wm"
        )

        rows = compute_local_moran_for_metric(
            analysis_rows,
            metric="normal_mean_wm",
            value_key="normal_mean_wm",
            radius_m=self.radius_m,
            window_width_m=self.window_width_m,
            weight_mode=self.weight_mode,
            permutations=self.permutations,
            seed=self.seed,
            alpha_raw=self.alpha_raw,
        )

        apply_bh_fdr(
            rows,
            alpha_fdr=self.alpha_fdr,
        )

        self._log_result_summary(
            rows,
            "normal_mean_wm",
        )

        return rows

    def _compute_alongshore(
        self,
        analysis_rows: list[dict],
    ) -> list[dict]:
        logger.info(
            "Computing Local Moran for {}",
            self.alongshore_metric,
        )

        rows = compute_local_moran_for_metric(
            analysis_rows,
            metric=self.alongshore_metric,
            value_key="alongshore_mean_wm",
            radius_m=self.radius_m,
            window_width_m=self.window_width_m,
            weight_mode=self.weight_mode,
            permutations=self.permutations,
            seed=self.seed,
            alpha_raw=self.alpha_raw,
        )

        apply_bh_fdr(
            rows,
            alpha_fdr=self.alpha_fdr,
        )

        self._log_result_summary(
            rows,
            self.alongshore_metric,
        )

        return rows

    @staticmethod
    def _log_result_summary(
        rows: list[dict],
        metric: str,
    ) -> None:
        """
        Записать распределение статусов и кластеров в лог.
        """
        status_counts = Counter(
            str(row["status"])
            for row in rows
        )

        raw_cluster_counts = Counter(
            str(row["cluster"])
            for row in rows
        )

        fdr_cluster_counts = Counter(
            str(row["cluster_fdr"])
            for row in rows
        )

        logger.info(
            "{} status distribution: {}",
            metric,
            dict(sorted(status_counts.items())),
        )

        logger.info(
            "{} raw cluster distribution: {}",
            metric,
            dict(sorted(raw_cluster_counts.items())),
        )

        logger.info(
            "{} FDR cluster distribution: {}",
            metric,
            dict(sorted(fdr_cluster_counts.items())),
        )

    @staticmethod
    def _validate_results(
        result_rows: list[dict],
        analysis_rows: list[dict],
    ) -> None:
        """
        Проверить результаты перед записью в SQLite.
        """
        expected_count = len(analysis_rows) * 2

        if len(result_rows) != expected_count:
            raise RuntimeError(
                "Unexpected result count: "
                f"expected={expected_count}, "
                f"actual={len(result_rows)}"
            )

        required_columns = {
            "point_id",
            "metric",
            "value",
            "n_window",
            "n_neighbors",
            "window_mean",
            "window_sd",
            "z_value",
            "spatial_lag",
            "local_i",
            "permutation_expected_i",
            "permutation_z",
            "p_sim",
            "q_fdr_bh",
            "quadrant",
            "cluster",
            "cluster_fdr",
            "status",
        }

        seen_keys: set[tuple[int, str]] = set()

        for row_number, row in enumerate(result_rows):
            missing = required_columns - set(row)

            if missing:
                raise RuntimeError(
                    "Local Moran result row is missing "
                    f"columns at row {row_number}: "
                    f"{sorted(missing)}"
                )

            key = (
                int(row["point_id"]),
                str(row["metric"]),
            )

            if key in seen_keys:
                raise RuntimeError(
                    "Duplicate Local Moran result key: "
                    f"{key}"
                )

            seen_keys.add(key)

            status = str(row["status"])

            if status == "OK":
                if row["p_sim"] is None:
                    raise RuntimeError(
                        "Successful result has no p_sim: "
                        f"{key}"
                    )

                if row["q_fdr_bh"] is None:
                    raise RuntimeError(
                        "Successful result has no "
                        f"q_fdr_bh: {key}"
                    )

                if row["local_i"] is None:
                    raise RuntimeError(
                        "Successful result has no local_i: "
                        f"{key}"
                    )

    def run(self) -> LocalMoranStats:
        """
        Выполнить расчёт и записать результаты в базу.

        Returns
        -------
        LocalMoranStats
            Статистика выполненного запуска.
        """
        self._validate_input_files()
        self._log_parameters()

        repository = LocalMoranRepository(
            self.db_path
        )

        logger.info(
            "Initializing Local Moran database schema"
        )
        repository.initialize()

        logger.info(
            "Reading points and trend metrics from {}",
            self.db_path,
        )

        source_rows = (
            repository.read_source_points_and_trends(
                self.alongshore_metric
            )
        )

        self._validate_source_rows(source_rows)

        logger.info(
            "Source points loaded: {}",
            len(source_rows),
        )

        logger.info(
            "Assigning points to coastline and "
            "computing chainages"
        )

        prepared = assign_points_to_coastline(
            source_rows=source_rows,
            coastline_geojson=self.coastline_geojson,
            metric_crs=self.metric_crs,
            max_snap_distance_m=(
                self.max_snap_distance_m
            ),
            strict_snap=self.strict_snap,
        )

        self._validate_prepared_geometry(prepared)

        logger.info(
            "Snap diagnostics: total={}, snap_ok={}, "
            "snap_bad={}, mean_snap_distance_m={:.3f}, "
            "max_snap_distance_found_m={:.3f}",
            prepared.n_total,
            prepared.n_snap_ok,
            prepared.n_snap_bad,
            prepared.mean_snap_distance_m,
            prepared.max_snap_distance_found_m,
        )

        analysis_rows = filter_snap_ok_rows(
            prepared.point_rows
        )

        if not analysis_rows:
            raise ValueError(
                "No points remain after snap-distance "
                "filtering"
            )

        if len(analysis_rows) != prepared.n_snap_ok:
            raise RuntimeError(
                "Snap filtering returned an unexpected "
                "number of points: "
                f"expected={prepared.n_snap_ok}, "
                f"actual={len(analysis_rows)}"
            )

        if prepared.n_snap_bad:
            logger.warning(
                "{} points will be stored in point-order "
                "diagnostics but excluded from Local Moran",
                prepared.n_snap_bad,
            )

        normal_rows = self._compute_normal(
            analysis_rows
        )

        alongshore_rows = self._compute_alongshore(
            analysis_rows
        )

        result_rows = (
            normal_rows
            + alongshore_rows
        )

        self._validate_results(
            result_rows,
            analysis_rows,
        )

        created_utc = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")

        coastline_hash = sha256_of_file(
            self.coastline_geojson
        )

        logger.info(
            "Writing Local Moran run metadata "
            "and results to database"
        )

        run_id = repository.insert_run(
            created_utc=created_utc,
            coastline_path=str(
                self.coastline_geojson
            ),
            coastline_sha256=coastline_hash,
            coastline_crs=prepared.coastline_crs,
            metric_crs=prepared.metric_crs,
            radius_m=self.radius_m,
            window_width_m=self.window_width_m,
            max_snap_distance_m=(
                self.max_snap_distance_m
            ),
            weight_mode=self.weight_mode,
            permutations=self.permutations,
            random_seed=self.seed,
            alpha_raw=self.alpha_raw,
            alpha_fdr=self.alpha_fdr,
            alongshore_metric=(
                self.alongshore_metric
            ),
            strict_snap=self.strict_snap,
            n_points_total=prepared.n_total,
            n_points_snap_ok=prepared.n_snap_ok,
            n_results=len(result_rows),
        )

        try:
            repository.insert_point_order(
                run_id,
                prepared.point_rows,
            )

            repository.insert_results(
                run_id,
                result_rows,
            )

        except Exception:
            logger.exception(
                "Failed to write Local Moran details "
                "for run_id={}. The run metadata may "
                "already exist in local_moran_runs.",
                run_id,
            )
            raise

        stats = LocalMoranStats(
            run_id=run_id,
            n_points_total=prepared.n_total,
            n_points_snap_ok=prepared.n_snap_ok,
            n_points_snap_bad=prepared.n_snap_bad,
            n_results=len(result_rows),
            output_db=self.db_path,
        )

        logger.success(
            "Local Moran completed: run_id={}, "
            "total_points={}, snap_ok={}, snap_bad={}, "
            "results={}, database={}",
            stats.run_id,
            stats.n_points_total,
            stats.n_points_snap_ok,
            stats.n_points_snap_bad,
            stats.n_results,
            stats.output_db,
        )

        return stats
