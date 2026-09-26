"""Сервис расчёта индексов волновой экспозиции (WEI / WER) для всей трассы.

Читает результаты шага 8:
    - ``wave_climate_daily.geojson``   — дневной ряд (одна запись = одна точка × день)
    - ``wave_climate_summary.geojson`` — сводная статистика (одна запись = одна точка)

Алгоритм
--------
1. Для каждой точки рассчитываются четыре частных показателя:
       mean_CWEF_Wm  — среднее CWEF [Вт/м]         → основа R1
       E_storm_MJm   — суммарная штормовая энергия  → основа R2
       K_dir         — коэффициент направленности   → основа R3
       CV            — коэффициент вариации         → основа R4

2. По ансамблю всех точек присваиваются квинтильные ранги R1–R4 ∈ [1, 5].

3. Итоговый индекс WER = (R1 · R2 · R3 · R4)^(1/4)     (1)

Выходной файл: ``wave_exposure_index.geojson`` — копия summary с добавленными
колонками mean_CWEF_Wm, E_storm_MJm, K_dir, CV, R1, R2, R3, R4, WER.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from src.waves.indices import WaveExposureIndex, WaveExposureRanker


class WaveExposureIndexService:
    """Батч-расчёт показателей WEI и рангов WER по всем точкам трассы.

    Parameters
    ----------
    storm_percentile : float
        Перцентиль CWEF — штормовой порог (по умолчанию 90).
    storm_threshold : float | None
        Явный порог [Вт/м]. Если задан — percentile игнорируется.
    dt_seconds : float
        Временной шаг ряда в секундах (86 400 для суточных данных).
    """

    def __init__(
        self,
        storm_percentile: float = 90.0,
        storm_threshold: float | None = None,
        dt_seconds: float = 86_400.0,
    ) -> None:
        self._calc = WaveExposureIndex(
            storm_percentile=storm_percentile,
            storm_threshold=storm_threshold,
            dt_seconds=dt_seconds,
        )
        self._ranker = WaveExposureRanker()

    # ── публичный API ─────────────────────────────────────────────────────────

    def compute_from_daily_gdf(
        self,
        daily_gdf: gpd.GeoDataFrame,
        summary_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Обогащает summary_gdf показателями WEI и ранговым индексом WER.

        Шаг 1: рассчитать показатели для каждой точки (WaveExposureIndex).
        Шаг 2: присвоить ранги R1–R4 и WER по ансамблю (WaveExposureRanker).

        Parameters
        ----------
        daily_gdf : gpd.GeoDataFrame
            Дневной ряд (шаг 8). Принимает CWEF как «CWEF_W_m» или «CWEF_Wm».
        summary_gdf : gpd.GeoDataFrame
            Сводная статистика (шаг 8). Одна строка на точку.

        Returns
        -------
        gpd.GeoDataFrame
            Копия summary_gdf + колонки:
            mean_CWEF_Wm, E_storm_MJm, K_dir, CV,
            R1, R2, R3, R4, WER.
        """
        # ── Шаг 1: показатели для каждой точки ───────────────────────────────
        daily_copy = daily_gdf.copy()
        daily_copy["_pid_str"] = daily_copy["point_id"].astype(str)

        index_records: list[dict] = []
        total = len(summary_gdf)
        for i, (_, row) in enumerate(summary_gdf.iterrows()):
            pid_str = str(row["point_id"])
            subset = daily_copy[daily_copy["_pid_str"] == pid_str].copy()
            indices = self._calc.compute(subset)
            indices["point_id"] = row["point_id"]
            index_records.append(indices)
            if (i + 1) % 25 == 0 or (i + 1) == total:
                logger.info(f"  Показатели рассчитаны: {i + 1}/{total}")

        if not index_records:
            logger.warning("Ни одна точка не дала показателей — возвращаем summary без изменений")
            return summary_gdf.copy()

        idx_df = pd.DataFrame(index_records)

        # Присоединяем показатели к summary
        summary_out = summary_gdf.copy()
        summary_out["_pid_str"] = summary_out["point_id"].astype(str)
        idx_df["_pid_str"] = idx_df["point_id"].astype(str)

        merged = summary_out.merge(
            idx_df.drop(columns=["point_id"]),
            on="_pid_str",
            how="left",
        ).drop(columns=["_pid_str"])

        # ── Шаг 2: ранжирование по ансамблю → R1..R4, WER ────────────────────
        logger.info("Присвоение рангов R1–R4 и расчёт WER по ансамблю точек...")
        ranked_gdf = self._ranker.rank_gdf(
            gpd.GeoDataFrame(merged, geometry="geometry", crs=summary_gdf.crs)
        )

        n_valid = ranked_gdf["WER"].notna().sum()
        logger.info(
            f"WER рассчитан для {n_valid}/{total} точек  "
            f"(min={ranked_gdf['WER'].min():.2f}, "
            f"max={ranked_gdf['WER'].max():.2f}, "
            f"mean={ranked_gdf['WER'].mean():.2f})"
        )
        return ranked_gdf

    def export(
        self,
        daily_geojson_path: str | Path,
        summary_geojson_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Читает файлы шага 8, рассчитывает WEI + WER, сохраняет GeoJSON.

        Parameters
        ----------
        daily_geojson_path : str | Path
            Путь к ``wave_climate_daily.geojson`` (шаг 8).
        summary_geojson_path : str | Path
            Путь к ``wave_climate_summary.geojson`` (шаг 8).
        output_path : str | Path
            Путь для записи ``wave_exposure_index.geojson``.

        Returns
        -------
        Path
            Абсолютный путь к сохранённому файлу.
        """
        daily_path   = Path(daily_geojson_path)
        summary_path = Path(summary_geojson_path)
        out_path     = Path(output_path)

        logger.info(f"Загрузка дневного ряда:       {daily_path}")
        daily_gdf = gpd.read_file(daily_path)

        logger.info(f"Загрузка сводной статистики:  {summary_path}")
        summary_gdf = gpd.read_file(summary_path)

        logger.info(
            f"Расчёт WEI+WER для {len(summary_gdf)} точек "
            f"({len(daily_gdf):,} дневных записей)"
        )
        result_gdf = self.compute_from_daily_gdf(daily_gdf, summary_gdf)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.to_file(out_path, driver="GeoJSON")
        logger.success(f"Индексы WEI+WER сохранены: {out_path}")
        return out_path
