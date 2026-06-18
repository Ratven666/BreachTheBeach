# 🌊 BreachTheBeach

**BreachTheBeach** — это реализация метода анализа волнового воздействия на береговую линию. 
Она объединяет обработку данных о береговой линии (OSM / GeoJSON), загрузку батиметрии (GEBCO, EMODnet), 
расчет ветрового разгона волн (fetch), интеграцию исторической метеорологии (Open-Meteo / ERA5) и 
итоговый расчет индекса волновой экспозиции (Wave Exposure Rank, **WER**).

---

## 📋 Содержание

- [Концепция](#-концепция)
- [Архитектура проекта](#-архитектура-проекта)
- [Пайплайн расчёта](#-пайплайн-расчёта)
- [Индекс волновой экспозиции WER](#-индекс-волновой-экспозиции-wer)
- [Источники данных](#-источники-данных)
- [Установка](#-установка)
- [Быстрый старт](#-быстрый-старт)
- [Структура файлов](#-структура-файлов)
- [Зависимости](#-зависимости)
- [Лицензия](#-лицензия)

---

## 🎯 Концепция

Платформа решает задачу **количественной оценки волновой уязвимости береговой линии**. 
Для каждой точки вдоль берега определяется:

- длина ветрового разгона волн (fetch) по заданным направлениям;
- история ветра и расчетная высота/период волн (метод SMB);
- трансформация волн на мелководье (рефракция, шолинг);
- итоговый поток волновой энергии (CWEF, Вт/м).

По ансамблю точек формируется ранговый индекс **WER** ∈ [1, 5], позволяющий сравнивать участки берега между собой по уровню волнового воздействия.

---

## 🏗 Архитектура проекта

Проект организован по принципу:

```
BreachTheBeach/
├── src/
│   ├── base/               # Вспомогательные типы (BBox и др.)
│   ├── coastline/          # Обработка береговой линии
│   │   ├── domain/         # Модели CoastlineDataset, CoastlinePointSet
│   │   ├── point_strategies/  # Стратегии расстановки точек (EqualRadiusStrategy)
│   │   ├── exporters/      # GeoJSON / GPKG экспортёры
│   │   └── services/       # MainCoastlineBuilder, CoastlineNormalService
│   ├── wind_fetch/         # Расчёт ветрового разгона волн
│   │   ├── WindFetchCalculator.py
│   │   ├── SequentialMultiDirectionFetchCalculator.py
│   │   ├── WindFetchParallelRunner.py
│   │   ├── CoastlineSpatialIndex.py
│   │   ├── WindFetchVisualizer.py
│   │   └── WindFetchConfig.py
│   ├── bathymetry/         # Загрузка и интерполяция батиметрии
│   │   ├── domain/         # GeoPoint, GeoLine, батиметрические модели
│   │   ├── loaders/        # GEBCO (OpenTopography), EMODnet
│   │   ├── interpolation/  # Интерполяция глубин по профилю
│   │   ├── exporters/      # GeoTIFF, NetCDF экспортёры
│   │   ├── cache/          # Кэширование запросов
│   │   └── services/       # BathymetryService
│   ├── weather_history/    # Историческая метеорология
│   │   ├── domain/         # WeatherLayerWrapper, WeatherPoint
│   │   ├── wheather_downloaders/  # Open-Meteo / ERA5 через cdsapi
│   │   ├── wind_rose/      # Роза ветров (matplotlib, plotly)
│   │   └── services/
│   ├── waves/              # Волновая климатология
│   │   ├── domain/         # Модели волнового климата
│   │   ├── offshore/       # Расчёт глубоководных волн (SMB)
│   │   ├── nearshore/      # Трансформация на мелководье
│   │   ├── energy.py       # WaveEnergyCalculator (CWEF)
│   │   ├── stats.py        # Статистика CWEF по точке
│   │   ├── indices.py      # WaveExposureIndex, WaveExposureRanker
│   │   ├── index_service.py
│   │   └── services/       # WaveClimateBatchProcessor
│   └── utils/              # Настройка логгера
├── pypeline/               # Последовательные шаги пайплайна (1–9)
├── main.py                 # Демо: привязка метеоданных и роза ветров
├── pyproject.toml
└── README.md
```

---

## 🔄 Пайплайн расчёта

Пайплайн состоит из 9 последовательных скриптов в директории `pypeline/`. Каждый шаг читает результаты предыдущего и сохраняет свои файлы в рабочую директорию (по умолчанию `nvrsk_calc/`).

### Шаг 1 — Построение береговой линии (`1_coastline_builder.py`)

Читает исходный GeoJSON с береговой линией и разделяет её на **главную** (`main_coastline.geojson`) и **вспомогательные** линии (`other_lines.geojson`). Применяет геометрическую очистку: snap-выравнивание вершин, обрезку коротких «листьев», фильтрацию по углу.

```python
builder = MainCoastlineBuilder(
    input_path="demo/NovorossCoastlineAdded.geojson",
    snap_tolerance_m=3.0,
    prune_leaf_length_m=80.0,
    prune_iterations=30,
    angle_tolerance_deg=1.0,
)
result = builder.build(save=True)
```

### Шаг 2 — Расстановка точек вдоль берега (`2_point_extraction.py`)

Равномерно расставляет точки через заданный шаг (например, 500 м или 1 км) вдоль главной береговой линии. Результат — GeoJSON с точками и их идентификаторами (`point_id`).

```python
extractor = CoastlinePointExtractor()
radius_points = extractor.extract(
    dataset=dataset,
    strategy=EqualRadiusStrategy(radius_step_m=500.0),
)
```

### Шаг 3 — Расчёт нормалей (`3_normals_calculation.py`)

Для каждой точки вычисляет азимут нормали к береговой линии. Нормаль указывает в сторону открытой акватории и используется для отбора «действующих» направлений волнения. Сохраняет поле `normal_azimuth_deg` в GeoJSON с точками.

### Шаг 4 — Загрузка метеосетки (`4_whether_point_colculation.py`)

Формирует регулярную сетку точек, покрывающую bbox береговых точек, и скачивает **историю ежедневных данных о ветре** (скорость `wind_speed_10m_max` и направление `wind_direction_10m_dominant`) через API Open-Meteo (модель ERA5, шаг сетки 0.25°). Данные кэшируются локально.

```python
config = WeatherDownloadConfig(
    model="era5",
    grid_step=0.25,
    daily_variables=("wind_speed_10m_max", "wind_direction_10m_dominant"),
)
weather_service = WeatherHistoryService(config=config)
```

### Шаг 5 — Слияние и подготовка береговой геометрии (`5_line_merger.py`)

Объединяет береговые линии в единые мультилинии (`merged_main.geojson`, `merged_other.geojson`), совместимые с форматом, ожидаемым модулем wind_fetch.

### Шаг 6 — Расчёт fetch (`6_ray_tracer.py`)

Запускает **трассировку лучей** из каждой береговой точки по заданным азимутам (0–360° с шагом 1° или по выбранным направлениям). Луч трассируется в WGS84 геодезической геометрии; остановка — при пересечении с береговой линией или при достижении максимальной длины (по умолчанию 100 км).

```python
config = WindFetchConfig(
    default_fetch_m=100_000.0,
    azimuths_deg=list(range(0, 360)),
)
calculator = SequentialMultiDirectionFetchCalculator(paths=paths, config=config)
results = calculator.calculate()
calculator.save_minimal(results, output_dir="nvrsk_calc/fetch")
```

Результат: `fetch_by_point.csv` и `fetch_by_point.geojson` — длина разгона по каждому направлению для каждой точки.

### Шаг 7 — Загрузка батиметрии (`7_bathymetry.py`)

Скачивает батиметрические данные для bbox акватории из двух источников:

| Источник | Разрешение | API |
|---|---|---|
| **GEBCO** (via OpenTopography) | ~450 м | OpenTopography REST API |
| **EMODnet** | ~115 м (для европейских акваторий) | OGC WCS |

Данные кэшируются в `bathymetry/cache/`. Поддерживается режим `"auto"` — автоматический выбор источника, а также принудительный выбор (`"gebco"` / `"emodnet"`). Батиметрия экспортируется в **GeoTIFF** и **NetCDF**.

```python
loader = BathymetryLoaderFactory.create(mode="auto", api_key=OP_TOP_KEY)
service = BathymetryService(loader=loader, cache=BathymetryCache(cache_dir))
```

### Шаг 8 — Расчёт волнового климата (`8_wave_climate_batch_pipeline.py`)

Основной вычислительный шаг. Для каждой береговой точки и каждого дня в периоде (например, 1940–2026):

1. По ветровым данным (скорость, направление) и длине fetch рассчитываются **глубоководные параметры волн** методом SMB (значимая высота `Hs_offshore`, период `Tp`).
2. По батиметрическому профилю в направлении волнения применяется **трансформация на мелководье** (рефракция и шолинг) → `Hs_nearshore`.
3. Рассчитывается **поток волновой энергии** (Coastal Wave Energy Flux):

```
P = ρ · g² · Hs² · Tp / (64π)      [Вт/м]
CWEF = P · cos(θ_wave − θ_normal)   [Вт/м]
```

Выходные файлы:
- `wave_climate_daily.geojson` — суточный ряд CWEF для каждой точки
- `wave_climate_summary.geojson` — сводная статистика (mean, p90, p99, max, E_storm и др.)

### Шаг 9 — Расчёт индекса WER (`9_wave_exposure_index_pipeline.py`)

Финальный шаг. По ансамблю всех точек присваиваются квинтильные ранги и рассчитывается итоговый **Wave Exposure Rank**. Результат сохраняется в `wave_exposure_index.geojson`.

---

## 📊 Индекс волновой экспозиции WER

**WER (Wave Exposure Rank)** — интегральный показатель волновой экспозиции точки берега ∈ [1, 5]. Вычисляется как геометрическое среднее четырёх частных рангов:

\[
\text{WER} = \left(R_1 \cdot R_2 \cdot R_3 \cdot R_4\right)^{1/4}
\]

Каждый ранг R ∈ {1, 2, 3, 4, 5} присваивается квинтильным методом по ансамблю всех точек трассы:

| Ранг | Показатель | Физический смысл |
|---|---|---|
| **R1** | Среднегодовое CWEF [Вт/м] | Фоновая (хроническая) волновая нагрузка |
| **R2** | Суммарная штормовая энергия `E_storm` [МДж/м] | Экстремальное воздействие (события выше p90) |
| **R3** | Коэффициент направленной концентрации `K_dir` | Преобладание одного направления волнения |
| **R4** | Коэффициент вариации `CV = σ / CWEF̄` | Межгодовая изменчивость волнового климата |

Поля в итоговом GeoJSON:
- `mean_CWEF_Wm`, `E_storm_MJm`, `storm_threshold_Wm`
- `K_dir`, `CV`, `n_days`, `n_storm_days`, `top3_sectors`
- `R1`, `R2`, `R3`, `R4`
- **`WER`** — итоговый ранг

---

## 🗄 Источники данных

| Тип данных | Источник | Формат входных данных |
|---|---|---|
| Береговая линия | Пользовательский GeoJSON (OSM и др.) | `.geojson` |
| Ветер (история) | Open-Meteo ERA5 (1940–present) | HTTP API → `.geojson` кэш |
| Батиметрия (грубая) | GEBCO via OpenTopography | REST API (требует `OP_TOP_KEY`) |
| Батиметрия (детальная) | EMODnet Bathymetry | OGC WCS |

Ключ API OpenTopography хранится в файле `secret/OPEN_TOPOGRAPHY_API.py`:

```python
# secret/OPEN_TOPOGRAPHY_API.py
OP_TOP_KEY = "ваш_ключ_здесь"
```

Бесплатный ключ доступен на [opentopography.org](https://opentopography.org).

---

## ⚙️ Установка

Проект использует [Poetry](https://python-poetry.org/) для управления зависимостями. Требуется **Python 3.12**.

```bash
# Клонировать репозиторий
git clone https://github.com/Ratven666/BreachTheBeach.git
cd BreachTheBeach
git checkout realese

# Установить зависимости через Poetry
poetry install

# Активировать виртуальное окружение
poetry shell
```

### Альтернатива: pip + venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Быстрый старт

### 1. Подготовьте данные

Поместите исходный GeoJSON с береговой линией в `demo/`:

```
demo/
├── NovorossCoastlineAdded.geojson   # береговая линия
└── ...
```

Создайте файл с ключом OpenTopography:

```bash
mkdir -p secret
echo 'OP_TOP_KEY = "ваш_ключ"' > secret/OPEN_TOPOGRAPHY_API.py
```

### 2. Запустите пайплайн пошагово

```bash
# Шаг 1: Построение береговой линии
python pypeline/1_coastline_builder.py

# Шаг 2: Расстановка точек (шаг 1000 м)
python pypeline/2_point_extraction.py

# Шаг 3: Расчёт нормалей
python pypeline/3_normals_calculation.py

# Шаг 4: Загрузка метеосетки ERA5 (1940–2026)
python pypeline/4_whether_point_colculation.py

# Шаг 5: Слияние береговых линий
python pypeline/5_line_merger.py

# Шаг 6: Расчёт fetch по 360 направлениям
python pypeline/6_ray_tracer.py

# Шаг 7: Загрузка батиметрии
python pypeline/7_bathymetry.py

# Шаг 8: Расчёт волнового климата
python pypeline/8_wave_climate_batch_pipeline.py

# Шаг 9: Расчёт индекса WER
python pypeline/9_wave_exposure_index_pipeline.py
```

### 3. Результаты

После выполнения всех шагов в директории `nvrsk_calc/` будут сохранены:

```
nvrsk_calc/
├── nvrsk_main_coastline.geojson
├── nvrsk_equal_radius_1000m_points_with_normals.geojson
├── points_with_weather.geojson
├── fetch/
│   ├── fetch_by_point.csv
│   └── fetch_by_point.geojson
├── bathymetry_from_bbox/
│   └── *.tif / *.nc
├── wave_climate_daily.geojson
├── wave_climate_summary.geojson
└── wave_exposure_index.geojson     ← итоговый результат
```

### 4. Демонстрация: роза ветров

```bash
python main.py
# → output/point_00000_wind_rose.png
# → output/point_00000_wind_rose.html
```

---

## 🌬 Роза ветров

Модуль `weather_history.wind_rose` строит розу ветров для любой точки метеосетки. Поддерживаются два бэкенда:

```python
# Matplotlib (PNG)
point.plot_wind_rose_matplotlib(
    output_path="output/wind_rose.png",
    nsector=64,
)

# Plotly (интерактивный HTML)
point.plot_wind_rose_plotly(
    output_path="output/wind_rose.html",
    nsector=64,
)
```

---

## 📦 Структура файлов

```
pypeline/                         # Скрипты пайплайна
├── 1_coastline_builder.py        # Шаг 1: береговая линия
├── 2_point_extraction.py         # Шаг 2: точки на берегу
├── 3_normals_calculation.py      # Шаг 3: нормали к берегу
├── 4_1_whether_grid_demo_to_qgis.py  # Шаг 4a: визуализация сетки
├── 4_whether_point_colculation.py    # Шаг 4b: загрузка метео
├── 5_line_merger.py              # Шаг 5: слияние линий
├── 6_ray_tracer.py               # Шаг 6: расчёт fetch
├── 7_bathymetry.py               # Шаг 7: батиметрия
├── 8_wave_climate_batch_pipeline.py  # Шаг 8: волновой климат
└── 9_wave_exposure_index_pipeline.py # Шаг 9: индекс WER

src/                              # Исходный код модулей
├── base/
├── coastline/
├── wind_fetch/
├── bathymetry/
├── weather_history/
├── waves/
└── utils/
```

---

## 📚 Зависимости

| Пакет | Назначение |
|---|---|
| `geopandas` | Геопространственные операции |
| `shapely` | Геометрические вычисления |
| `pyproj` | Геодезические трансформации и проекции |
| `rasterio` | Чтение/запись растровых данных (GeoTIFF) |
| `numpy`, `scipy` | Научные вычисления, интерполяция |
| `pandas` | Работа с временными рядами |
| `xarray`, `netcdf4` | Работа с NetCDF (ERA5, батиметрия) |
| `cdsapi` | Загрузка данных ERA5 через CDS API |
| `cfgrib` | Чтение GRIB-файлов ERA5 |
| `osmium` | Обработка данных OpenStreetMap |
| `networkx` | Топологический анализ береговых линий |
| `matplotlib`, `plotly` | Визуализация (роза ветров, графики) |
| `requests` | HTTP-запросы к API |
| `pydap` | Доступ к OPeNDAP-серверам (EMODnet) |
| `loguru` | Структурированное логирование |

Полный список версий — в `pyproject.toml` / `poetry.lock`.

---

## 📐 Научная основа

Методологическая база платформы:

- **SMB (Sverdrup–Munk–Bretschneider)** — эмпирический метод расчёта волн по ветровым данным и длине разгона.
- **Shoaling & Refraction** — трансформация волн при распространении на мелководье.
- **CWEF (Coastal Wave Energy Flux)** — проекция потока волновой энергии на нормаль к берегу, принятая как мера воздействия.
- **WER** — квинтильное ранжирование по 4 независимым критериям (Harley 2017 и др.).

---

## 📄 Лицензия

Distributed under the MIT License. See `LICENSE` for more information.

### Ссылка для цитирования 

```bibtex
@software{vystrchil2026breachthebeach,
  author    = {Vystrchil, Mikhail},
  title     = {{BreachTheBeach}: A Coastal Wave Exposure Analysis Platform},
  year      = {2026},
  publisher = {GitHub},
  version   = {0.1.0},
  url       = {https://github.com/Ratven666/BreachTheBeach},
  note      = {MIT License},
}
```
**APA:**

> Vystrchil, M. (2026). *BreachTheBeach: A Coastal Wave Exposure Analysis
> Platform* (Version 0.1.0) [Software]. GitHub.
> https://github.com/Ratven666/BreachTheBeach
---

## 👤 Автор

**Vystrchil MG** — [GitHub Profile](https://github.com/Ratven666)
