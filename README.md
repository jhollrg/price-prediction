# NYC Taxi Fare Prediction

Предсказание стоимости поездки такси (`fare_amount`) на данных NYC TLC Yellow Taxi
с помощью SQL-пайплайна для фичей, SHAP-based отбора признаков и сравнения
классических ML-моделей.

## Задача

По координатам посадки/высадки, времени и служебным полям поездки предсказать
стоимость поездки. Цель проекта — не просто обучить модель, а построить
воспроизводимый end-to-end пайплайн: сырые данные → фичи в БД → отбор
признаков → сравнение моделей → лучшая модель для инференса.

## Данные

- Источник: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) —
  публичный датасет поездок Yellow Taxi.
- По умолчанию используется `yellow_tripdata_2015-01.parquet` — последний год,
  где ещё есть координаты pickup/dropoff (с 2016 TLC перешёл на zone ID).
- Данные **не закоммичены** в репозиторий из-за размера (~12M строк, файлы —
  сотни МБ). Скачиваются одной командой:
  ```bash
  make data       # scripts/download_data.py — качает parquet в data/raw/
  make db         # строит SQLite и прогоняет sql/feature_queries.sql
  ```

## Пайплайн

```
data/raw/*.parquet
        │  scripts/download_data.py
        ▼
   SQLite: trips
        │  sql/create_tables.sql + sql/feature_queries.sql
        ▼
   SQLite: features  ──► src/data_loader.py ──► train/test split
        │
        ▼
 SHAP feature selection (CatBoost surrogate, src/feature_selection.py)
        │
        ▼
 LinearRegression → RandomForest → CatBoost → PyTorch MLP
        │
        ▼
   champion model (лучший RMSE) → models/champion.joblib
```

## Признаки

- **Temporal**: час/день недели, weekend/rush-hour флаги, циклическое
  sin/cos-кодирование часа и дня недели (без разрыва 23→0)
- **Geo**: haversine-расстояние между pickup/dropoff, бакеты дистанции
  (short / medium / long / very_long)
- **Zone-агрегаты**: средняя цена по зоне, число поездок в зоне, средняя
  скорость по зоне-часу (прокси загруженности)

Фичи считаются дважды — в SQL (`sql/feature_queries.sql`) для батч-обработки
и в Python (`src/features.py`) для онлайн-инференса по одной поездке; тесты
(`tests/test_features.py`) проверяют, что оба пути дают одинаковый результат.

## Модели и метрики

| Модель | Роль |
|---|---|
| LinearRegression | baseline (без тюнинга) |
| RandomForestRegressor | классический ансамбль |
| CatBoostRegressor | градиентный бустинг + native categorical, early stopping |
| PyTorch MLP | 256 → ReLU → Dropout → 128 → ReLU → 1 |

Метрики: **RMSE, MAE, R²** на отложенной выборке. Модель с лучшим RMSE
сохраняется как «чемпион» (`models/champion.joblib`) и используется в
`src/predict.py`.

Отбор признаков — SHAP (`get_feature_importance(type='ShapValues')` из
CatBoost, без внешней зависимости от пакета `shap` для расчёта значений).

## Структура проекта

```
notebooks/   EDA (eda.ipynb)
scripts/     download_data.py — скачивание сырых parquet
sql/         create_tables.sql, feature_queries.sql — схема и фичи
src/         config.py, data_loader.py, features.py,
             feature_selection.py, mlp_model.py, train.py, predict.py
tests/       pytest: test_features.py, test_predict.py
```

## Как запустить

```bash
make setup      # venv + зависимости из requirements.txt
make data       # скачать сырые parquet
make db         # построить SQLite и посчитать фичи
make train      # обучить все модели, сохранить чемпиона
make test       # прогнать тесты
```

Инференс на новых поездках:
```bash
python -m src.predict trips.csv --output_csv predictions.csv
```

## Стек

Python, pandas, NumPy, scikit-learn, CatBoost, PyTorch, SQLite/SQL,
SHAP (через CatBoost), pytest, black/isort.

Подробнее см. в файлах проекта
