# Assignment 4 — Практики курса на новых датасетах

Практические работы курса, повторённые на трёх открытых датасетах, которых не было на занятиях. В каждом ноутбуке —
рабочий код, результаты выполнения и краткие выводы. Сдаются английские ноутбуки; русские копии для чтения — `*_ru.ipynb`.

| № | Практика | Датасет | Ноутбук | Главный результат (test) |
|---|---|---|---|---|
| 1 | Подготовка данных и EDA | Hotel Booking Demand | `P1_data_preparation.ipynb` | найдены и удалены 5 столбцов-утечек; 119 208 броней × 34 признака, разбиение по времени |
| 2 | Baseline и метрики | Hotel Booking Demand | `P2_baseline_metrics.ipynb` | логистическая регрессия ROC-AUC 0,849, PR-AUC 0,806; случайная CV завышает (0,907 против 0,849) |
| 3 | Сравнение моделей и подбор гиперпараметров | Hotel Booking Demand | `P3_model_comparison_tuning.ipynb` | LightGBM PR-AUC 0,817 против 0,806; подбор заметно помогает только дереву решений (+0,222) |
| 4 | Transfer learning | Beans (фото листьев) | `P4_transfer_learning.ipynb` | замороженная EfficientNet-B0 + линейный слой 91,4 %; на 10 % данных замороженная ResNet-18 80,5 % против 71,1 % у CNN с нуля |
| 5 | Fine-tuning | Beans (фото листьев) | `P5_fine_tuning.ipynb` | дообученная ResNet-18 96,9 % против 89,8 % у замороженной |
| 6 | LLM + LoRA | Twitter Financial News | `P6_llm_lora.ipynb` | LoRA на Qwen2.5-0.5B: macro-F1 0,869 против 0,715 у замороженной LLM + линейного слоя и 0,336 у zero-shot |

## Датасеты

| Датасет | Размер | Задача | Источник, лицензия |
|---|---|---|---|
| Hotel Booking Demand | 119 390 броней, 32 столбца, 2015–2017 | бинарная: отменят ли бронь | Antonio, Almeida & Nunes, *Data in Brief* 22 (2019), CC BY 4.0; CSV из репозитория TidyTuesday |
| Beans | 1 295 фото листьев, 3 класса | классификация изображений: угловатая пятнистость / ржавчина / здоровый | Makerere AI Lab (2020), MIT; Hugging Face `AI-Lab-Makerere/beans` |
| Twitter Financial News Sentiment | 11 931 твит, 3 класса | классификация текста: медвежий / бычий / нейтральный | zeroshot (2022), MIT; Hugging Face `zeroshot/twitter-financial-news-sentiment` |

Каждый ноутбук при первом запуске сам скачивает данные в `data/`.

## Как запустить

Окружение: `requirements.txt` репозитория плюс `torchvision`. Запускать из этой папки.

* **P1 → P2 → P3** (CPU, по несколько минут): P1 пишет `data/hotels_clean.parquet`, его читают P2 и P3.
* **P4 → P5** (GPU; на Apple MPS около 10 минут каждый и тяжело для ноутбука — лучше GPU Kaggle): P5 читает
  `results/p4_summary.json` для таблицы сравнения.
* **P6** нужен GPU: `python kaggle_p6.py && kaggle kernels push -p kaggle/a4_p6` запускает его на бесплатной T4 в Kaggle;
  выполненный ноутбук забирается командой `kaggle kernels output azizabdulin/a4-p6-llm-lora -p kaggle/a4_p6/output`.

Ноутбуки генерируются из `build_p*.py` (markdown на английском и русском, общий код); `finalize.py` обновляет markdown
выполненных ноутбуков и пишет русские копии без повторного запуска. P4–P6 используют PyTorch и не импортируют
scikit-learn (две копии OpenMP падают на macOS); метрики для них — в `a4_metrics.py`, сверены со scikit-learn в
`tests/test_a4_metrics.py`.
