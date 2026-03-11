# Кредитный скоринг

Проект: оценка вероятности дефолта заёмщика и расчёт кредитного скора.

---

## Структура

```
DA_proj/
│
├──  train.ipynb          ← Обучение модели (запускать в Jupyter)
├──  app.py               ← Веб-интерфейс на Streamlit
├── requirements.txt     ← Зависимости Python
│
├── Данные:
│   ├── cs-training.csv     ← Обучающая выборка (150 000 строк)
│   └── cs-test.csv         ← Тестовая выборка
│
└── Артефакты модели (генерируются ноутбуком):
    ├── model.pkl            ← Обученная модель LogisticRegression
    ├── woe_mapping.csv      ← WoE-значения для каждого бина
    ├── bin_edges.json       ← Границы бинов для каждого признака
    ├── feature_cols.json    ← Список признаков модели
    └── scorecard.csv        ← Скорбалл (таблица очков)
```

---

## Запуск

### 1. Установить зависимости
```bash
pip install -r requirements.txt
```

### 2. Обучить модель (если нет артефактов)
Открыть и запустить `train.ipynb` в Jupyter

### 3. Запустить веб-приложение
```bash
streamlit run app.py
```
Откроется в браузере: http://localhost:8501

---

## 🧠 Модель

| Параметр | Значение |
|----------|----------|
| Алгоритм | Logistic Regression + WoE encoding |
| AUC-ROC | 0.8268 |
| Accuracy | 93.68% |
| Горизонт прогноза | 2 года |

### Признаки модели (7 штук):
| Признак | Описание |
|---------|----------|
| `RevolvingUtilizationOfUnsecuredLines` | % использования лимита по картам |
| `age` | Возраст заёмщика |
| `NumberOfTime30-59DaysPastDueNotWorse` | Просрочек 30–59 дней |
| `NumberOfTime60-89DaysPastDueNotWorse` | Просрочек 60–89 дней |
| `NumberOfTimes90DaysLate` | Просрочек 90+ дней |
| `MonthlyIncome` | Ежемесячный доход |
| `DebtRatio` | Долговая нагрузка (платежи / доход) |

### Формула скора:
$$Score = 650 + 72.13 \times \ln\left(\frac{1 - P_{default}}{P_{default}}\right)$$

Скор ограничен диапазоном **300–850**.

---
## Датасет

**Give Me Some Credit** (Kaggle)  
- 150 000 строк, реальные данные американских заёмщиков  
- Целевая переменная: `SeriousDlqin2yrs` — допустил ли просрочку 90+ дней за 2 года  
- Дисбаланс классов: ~93% не дефолт, ~7% дефолт

---

## Пайплайн

```
cs-training.csv
      ↓
  train.ipynb
  (EDA → WoE-кодирование → обучение → оценка)
      ↓
  model.pkl + woe_mapping.csv + bin_edges.json + ...
      ↓
    app.py
  (ввод данных → скор → рекомендации по кредиту)
```
