# Распознавание рукописного текста на кириллице: сбор датасета и тестирование моделей

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)

---

## Участники

- Калякин Тимофей Сергеевич
- Зубов Илья Леонидович
- Фахретдинов Муса Шамилевич
- Климов Игорь Игоревич
- Агапов Константин Сергеевич

---

## О проекте

Данный репозиторий является решением 1 кейса практикума 3 семестра магистратуры МФТИ.

Цель: создать и протестировать датасет для распознавания рукописного русского текста, который сможет стать частью системы автоматического парсинга данных из документов.

В репозитории:
- ссылка на датасет и правила его формирования,
- модульный пайплайн аугментаций для HTR/OCR,
- ноутбуки с экспериментами и бенчмарками моделей.

---

## Задачи

1. Провести обзор существующих открытых источников данных, подходов к аугментации, моделей;
2. Собрать единый датасет из разных источников;
3. Создать пайплайн аугментации;
4. Провести оценку существующих моделей;
5. Организовать репозиторий с результатами работы и выводами.

---

## Технологии

- **ML/HTR**: `torch`, `torchvision`, `tensorflow`, `keras`
- **Аугментации**: `albumentations`, `augraphy`
- **CV**: `opencv-python`, `scikit-image`
- **Данные и метрики**: `numpy`, `pandas`, `scipy`, `scikit-learn`
- **Эксперименты**: `mlflow`, `dvc`
- **Работа с датасетами**: `datasets` (используется в ноутбуках/примерах)

---

## Структура проекта

```
cyrill/
├── examples/
│   └── augmentation.ipynb           # Пример использования аугментаций
├── models/
│   ├── trocr_cyrillic.ipynb         # page-level бенчмарк
│   └── trocr_small.ipynb            # fine-tuning TrOCR-small
├── src/                             # Директория с аугментациями
│   └── preprocessing/
│       ├── augmentation_pipeline.py
│       ├── configs.py
│       ├── loaders/
│       ├── transforms/
│       ├── utils/
│       └── writers/
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Установка и запуск

### 1. Клонирование

```bash
git clone https://github.com/TimofeyKaliakin/cyrill.git 
cd cyrill
```

### 2. Установка зависимостей

```bash
brew install uv # MacOS
pip install uv # Wind
uv sync
source .venv/bin/activate
```

### 4. Проверка окружения

```bash
python main.py
```

---

## Подробно по разделам

Подробнее по каждому этапу работы можно ознакомиться в блоках ниже.


<details>
<summary><b><h2> Dataset </h2></b></summary>

Собранный датасет можно увидеть на https://huggingface.co/datasets/Timka28/cyrillic_small

---

## Общая статистика

- **Всего образцов:** 37 831  
- **Итоговые сплиты:** train — 32 299, test — 5 532  
- **Поля:**  
  - `image` — изображение (datasets.Image / Pillow)  
  - `text` — транскрипция  
  - `dataset` — источник  
  - `split` — итоговый сплит (train/test)  
  - `source_split` — сплит в исходном датасете (если был)

---

## Источники данных

| Dataset | Samples | Comment | Link |
|--------|---------|---------|---------|
| cyrillic-handwriting-dataset | 20 000 | Подвыборка | https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset |
| handwritten_ru_letters | 5 000 | Подвыборка | https://www.kaggle.com/datasets/olgabelitskaya/handwritten-russian-letters |
| comnist | 5 000 | Подвыборка | https://www.kaggle.com/datasets/gregvial/comnist |
| synthetic_cyrillic_large_00_02 | 5 000 | Подвыборка из первой папки | https://huggingface.co/datasets/pumb-ai/synthetic-cyrillic-large |
| school_notebooks | 1 857 | Взят полностью | https://huggingface.co/datasets/ai-forever/school_notebooks_RU/tree/main |
| HWR200 | 816 | Взяты папки 0-4 | https://huggingface.co/datasets/AntiplagiatCompany/HWR200 |
| handwritten_essay | 158 | Взят полностью | https://data.mendeley.com/datasets/vs44v8r3nf/1 |

---

## Правила формирования train/test

- Сохраняем исходные сплиты: handwritten_essay (train/test), cyrillic-handwriting-dataset (train/test), school_notebooks (val объединён в test).  
- HWR200: папка `4` → test, остальные папки → train.  
- comnist, handwritten_ru_letters, synthetic_cyrillic_large_00_02: случайное разбиение 70/30 (сид 42).  
`split` хранит итоговое разбиение, `source_split` — исходное.

---

## Статистика текста

- **Min длина:** 1 символ  
- **Max длина:** 3 730 символов  
- **Средняя:** 87.1  
- **Медиана:** 7  

---

## Пример записи

```json
{
  "image": "<image_bytes>",
  "text": "пример",
  "dataset": "cyrillic-handwriting-dataset",
  "split": "train",
  "source_split": "train"
}
```

---

## Как скачать с HF

1. `pip install -U datasets huggingface_hub`
2. Загрузить публичный датасет:
   ```python
   from datasets import load_dataset

   ds_dict = load_dataset("Timka28/cyrillic_small") 
   # или
   ds_train = load_dataset("Timka28/cyrillic_small", split="train")
   ds_test = load_dataset("Timka28/cyrillic_small", split="test")
   ```
Доступ открыт публично.

</details>


<details>
  <summary><b><h2> Image Augmentation Pipeline (OCR / Handwritten Text) </h2></b></summary>

## Введение

Данный модуль реализует модульный и расширяемый пайплайн аугментаций изображений, ориентированный на задачи:
- OCR
- распознавание рукописного текста
- устойчивость моделей к шумам сканирования и копирования

## Общая архитектура

- Каждое изображение может быть обработано не более чем одной аугментацией
- Аугментации выбираются вероятностно
- Все параметры аугментаций:
  - семплируются внутри аугментации
  - возвращаются в meta для логирования
- Пайплайн совместим с:
  - HuggingFace datasets
  - torchvision.transforms
  - инференсом в transformers

## Быстрый старт

### PipelineConfig, AugmentationPipeline и пример использования

`PipelineConfig` задаёт правила выбора аугментаций (какую применять и как часто).

```python
config = PipelineConfig(
    p_aug=0.9,
    augmentations={...},
    aug_weights={...},
    return_params=True,
)
```

**Поля:**
- `p_aug: float`  
  Вероятность вообще применить аугментацию к изображению.  
  Пример: 0.9 → 90% изображений будут аугментированы, 10% — останутся как есть.
- `augmentations: dict[str, BaseAugmentation]`  
  Словарь {name: augmentation_instance}.  
  Важно:
  - ключ name — это идентификатор аугментации в пайплайне
  - значения — инстансы ваших классов (ScaleAugmentation, GridDistortionAugmentation, …)
- `aug_weights: dict[str, float]`  
  Вероятности выбора каждой аугментации (если аугментация применяется).  
  Требование:
  - ключи должны 1к1 совпадать с augmentations
  - сумма весов должна быть равна 1.0 (в пределах float-погрешности)
- `return_params: bool`  
  Если True, то в meta будут записаны реально применённые параметры аугментации.  
  Это удобно для логирования и последующей аналитики.

---

`AugmentationPipeline` применяет не более одной аугментации на изображение.

```python
aug_pipeline = AugmentationPipeline(config=config, seed=42)
```

- `seed` (опционально) — чтобы выбор аугментации был детерминированным для одного и того же idx. Это важно для:
  - воспроизводимости экспериментов
  - стабильного поведения на HF датасете при повторных прогонах

---

**Что возвращает пайплайн:**

```python
img_aug, meta = aug_pipeline(img, idx=idx)
```

`meta` имеет структуру:
- если аугментация не применялась:
  ```python
  {"applied": False}
  ```
- если аугментация применялась:
  ```python
  {
    "applied": True,
    "name": "<ключ из augmentations>",
    "params": {...}   # только если return_params=True
  }
  ```

## Интеграция с HuggingFace datasets

### Почему нужен idx

Если ты используешь `seed`, то `idx` нужен, чтобы:
- случайность была “привязана” к индексу
- одно и то же изображение при повторном вызове давало один и тот же выбор аугментации (если хочешь воспроизводимость)

### HF transform-функция (batch → batch)

`with_transform` ожидает функцию, которая:
- принимает batch-словарь списков
- возвращает batch-словарь списков

```python
# Здесь вам уже нужно подготовить обработку под свою модель
# но главное используйте aug_pipeline(img, idx) передавая idx
def hf_transform(batch):
    images = []
    aug_metas = []

    for img, idx in zip(batch["image"], batch["idx"]):
        img_aug, meta = aug_pipeline(img, idx=idx) # Применяем аугментационный пайплайн

        images.append(img_aug)
        aug_metas.append(meta)

    batch["image"] = images
    batch["aug_meta"] = aug_metas
    return batch
```

### Применение к датасету

```python
ds_aug = ds.with_transform(
    hf_transform,
    columns=["image", "idx"],         # чтобы в batch точно были эти поля
    output_all_columns=True,          # чтобы не потерять остальные колонки
)
```

## Пример использования 
[Link](https://github.com/TimofeyKaliakin/cyrill/blob/master/examples/augmentation.ipynb)

Дальше ты работаешь с `ds_aug` как обычно:
- `ds_aug[i]` вернёт уже обработанное изображение и aug_meta
- трансформация применяется на лету при доступе к элементам

## BaseAugmentation

Все аугментации наследуются от `BaseAugmentation`.

Каждая аугментация:
- является callable-объектом
- реализует:
  - `sample_params()` — генерация параметров
  - `apply(image, params)` — применение аугментации
- возвращает:
  ```python
  (image, params)
  ```

## Описание аугментаций

### ScaleAugmentation

**Назначение:**  
Имитация изменения масштаба текста (разное расстояние до камеры / сканера).


**Параметры:**
```python
ScaleAugmentation(
    scale_range=(0.5, 1.0)
)
```
- `scale_range` — Диапазон масштабирования. Значения > 1 не рекомендуются.

---

### ShearAugmentation

**Назначение:**  
Геометрический сдвиг (наклон текста).

**Когда использовать:**  
- рукописный текст
- сканы под углом
- фотографии документов

**Параметры:**
```python
ShearAugmentation(
    shear_x_range=(-8.0, 8.0),
    shear_y_range=None
)
```
- `shear_x_range` — Диапазон углов shear по X (в градусах)
- `shear_y_range` — Диапазон углов shear по Y

> ⚠️ Должен быть задан хотя бы один из параметров.

---

### GridDistortionAugmentation

**Назначение:**  
Локальные геометрические искажения (неровная бумага, деформация скана).

**Реализация:**  
На базе albumentations.GridDistortion.

**Параметры:**
```python
GridDistortionAugmentation(
    num_steps_range=(3, 6),
    distort_limit_range=(0.1, 0.3)
)
```
- `num_steps_range` — Количество ячеек сетки
- `distort_limit_range` — Амплитуда локальных искажений

---

### ElasticTransformAugmentation

**Назначение:**  
Эластичные деформации символов (вариативность почерка).

**Реализация:**  
На базе albumentations.ElasticTransform.

**Параметры:**
```python
ElasticTransformAugmentation(
    alpha_range=(0.5, 2.0),
    sigma_range=(30.0, 60.0)
)
```
- `alpha_range` — Сила деформации
- `sigma_range` — Степень сглаживания

---

### MotionBlurAugmentation

**Назначение:**  
Имитация смаза при движении камеры.

**Когда использовать:**  
- фото документов
- мобильный OCR

**Параметры:**
```python
MotionBlurAugmentation(
    blur_limit_range=(3, 9),
    angle_range=(0, 360),
    direction_range=(-1.0, 1.0)
)
```
- `blur_limit_range` — Размер ядра размытия
- `angle_range` — Угол смаза
- `direction_range` — Направленность смаза

---

### DilationAugmentation

**Назначение:**  
Утолщение символов (жирная печать, расплывшиеся чернила).

**Параметры:**
```python
DilationAugmentation(
    kernel_size_range=(3, 5),
    iterations_range=(1, 2)
)
```

---

### ErosionAugmentation

**Назначение:**  
Истончение символов (плохая печать, выцветшие чернила).

**Параметры:**
```python
ErosionAugmentation(
    kernal_size_range=(3, 5),
    iterations_rnage=(1, 2)
)
```

---

### ScribblesAugmentation

**Назначение:**  
Случайные линии, пометки, надписи поверх текста.

**Реализация:**  
На базе augraphy.Scribbles.

**Параметры:**
```python
ScribblesAugmentation(
    size_range=(400, 600),
    count_range=(1, 6),
    thickness_range=(1, 3),
    brightness_values=(32, 64, 128),
    rotation_range=(0, 360)
)
```

---

### WaterMarkAugmentation

**Назначение:**  
Добавление водяных знаков (COPY, DRAFT, SAMPLE).

**Реализация:**  
На базе augraphy.WaterMark.

**Параметры:**
```python
WaterMarkAugmentation(
    words=("COPY", "DRAFT"),
    font_size_range=(10, 20),
    font_thickness_range=(1, 3),
    rotation_range=(0, 360)
)
```

---

### BadPhotoCopyAugmentation

**Назначение:**  
Имитация артефактов плохого сканирования или ксерокопирования
(полосы, шум, неравномерная заливка, дефекты печати).

**Когда использовать:**  
- сканы документов
- ксерокопии
- низкокачественные PDF
- OCR в реальных условиях

**Реализация:**  
На базе augraphy.BadPhotoCopy.

**Параметры:**
```python
BadPhotoCopyAugmentation(
    noise_type_range=(0, 3),
    noise_iteration_range=(1, 3),
    noise_size_range=(1, 3),
    noise_sparsity_range=(0.1, 0.5),
    noise_concentration_range=(0.1, 0.5),
)
```

- `noise_type_range` — диапазон типов шума
- `noise_iteration_range` — количество итераций шума
- `noise_size_range` — размер шумовых элементов
- `noise_sparsity_range` — разреженность шума
- `noise_concentration_range` — концентрация шума

## Логирование и отладка

Если `return_params=True`, пайплайн возвращает:

```python
{
  "applied": True,
  "name": "grid_distortion",
  "params": {...}
}
```

Эти данные можно сохранять в CSV для анализа качества.

---

## Полезные ссылки

- **Albumentations** — https://albumentations.ai
  - [GridDistortion](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.GridDistortion)
  - [ElasticTransform](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ElasticTransform)
  - [MotionBlur](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.MotionBlur)
- **Augraphy**
  - [BadPhotoCopy](https://augra.phy.readthedocs.io/en/latest/auggraphy.augmentations.badphotocopy.html)
  - [Scribbles](https://augra.phy.readthedocs.io/en/latest/auggraphy.augmentations.scribbles.html)
  - [WaterMark](https://augra.phy.readthedocs.io/en/latest/auggraphy.augmentations.watermark.html)
- **HuggingFace Datasets** — [`with_transform`](https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.Dataset.with_transform)
- **OpenCV** — [morphology](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)

---

> Документация ориентирована на задачи OCR и распознавания рукописного текста.

</details>


<details>
<summary><b><h2> Models </h2></b></summary>

## Методика экспериментов

### Файнтюнинг TrOCR-small

В ноутбуке `trocr_small.ipynb` выполнен файнтюнинг модели `microsoft/trocr-small-handwritten` на датасете `Timka28/cyrillic_small`: внутри `train` сделан сплит 80/20 (`seed=42`), изображения приводятся к RGB и проходят on-the-fly аугментации с `p_aug=0.5` (bad_photo, dilation, elastic_transform, erosion, grid_distortion, motion_blur, scale, scribbles). Тексты токенизируются с `MAX_TARGET_LENGTH=512`, обучение запускается через `Seq2SeqTrainer` в `fp16` при `lr=5e-5`, `epochs=15`, `batch_size=8`, `generation_max_length=512`. Протестирована `microsoft/trocr-small-handwritten`. 

### Бенчмарк TrOCR (page-level)

В ноутбуке `trocr_cyrillic.ipynb` реализован page-level бенчмарк на `Timka28/cyrillic` (`test`): изображения приводятся к RGB, при необходимости применяется `crop_ink`, после чего страница сегментируется на строки через бинаризацию и горизонтальную проекцию; строки распознаются батчами и склеиваются в текст страницы. Инференс выполняется с `num_beams=1`, `max_new_tokens=64`, `fp16` на GPU при наличии; метрики CER/WER считаются после нормализации пробелов, скорость фиксируется как `page_per_sec`. Протестированы `cyrillic-trocr/trocr-handwritten-cyrillic`, `kazars24/trocr-base-handwritten-ru`, `akushsky/trocr-large-handwritten-ru`.

## Результаты тестирования

| Модель | Настройки модели | CER | WER | `page_per_sec` |
|---|---|---|---|---|
| microsoft/trocr-small-handwritten | 0.06B | 0.81 | 0.99 | 2.143 |
| cyrillic-trocr/trocr-handwritten-cyrillic | 0.3B | 0.285 | 0.579 | 0.143 |
| kazars24/trocr-base-handwritten-ru | 0.3B | 0.617 | 0.938 | 0.176 |
| akushsky/trocr-large-handwritten-ru | 0.6B | 0.985 | 1.25 | 0.159 |

</details>

## Общие выводы по работе

- Собран и опубликован датасет рукописной кириллицы с воспроизводимыми правилами сплитов; он пригоден для обучения и объективного сравнения моделей.
- Реализован модульный пайплайн аугментаций, совместимый с HF datasets и torchvision, что упрощает эксперименты и имитацию реальных артефактов сканов.
- Для page-level HTR показан рабочий подход: сегментация страницы на строки, построчный инференс и сборка в текст страницы с оценкой CER/WER.
- Лучшее качество по CER/WER показывает `cyrillic-trocr/trocr-handwritten-cyrillic`, но это самая медленная модель по `page_per_sec`.
- `microsoft/trocr-small-handwritten` работает заметно быстрее остальных, но качество распознавания существенно хуже; рост размера модели не гарантирует улучшения метрик.
