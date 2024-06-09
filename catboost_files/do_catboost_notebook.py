import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
import os

# Путь к файлу
file_path = "Анализ стратегий конкурентов.xlsx"

# Загрузка данных с корректным указанием начала меток
data = pd.read_excel(file_path, header=2)

# Извлечение текстов новостей и меток
texts = data.iloc[:, 28]  # Предполагается, что тексты находятся в столбце AC (29-й столбец, индекс 28)
tags = data.iloc[:, 7:29]  # Метки находятся в столбцах H-AB (8-29 столбцы)

# Преобразование меток
tags = tags.fillna(0).replace("+", 1)

# Удаление строк с NaN значениями в текстах и метках
combined = pd.concat([texts, tags], axis=1).dropna()
texts = combined.iloc[:, 0]
tags = combined.iloc[:, 1:]

# Подготовка данных для моделей
dfs = [pd.DataFrame({"text": texts, "label": tags.iloc[:, i]}) for i in range(tags.shape[1])]
dfs_split = [train_test_split(df["text"], df["label"], test_size=0.2, random_state=42) for df in dfs]

# Проверка распределения меток в обучающей и тестовой выборках
for idx, (X_train, X_test, y_train, y_test) in enumerate(dfs_split):
    print(f"Категория {idx}:")
    print("Распределение меток в y_train:")
    print(y_train.value_counts())
    print("Распределение меток в y_test:")
    print(y_test.value_counts())

# Если необходимо, вручную добавьте строки с недостающими классами в y_train и y_test

# Функция для обучения модели CatBoost
def fit_catboost(X_train, X_test, y_train, y_test, catboost_params={}, verbose=100):
    # Проверка данных перед созданием пулов
    print(f"\nПроверка данных для обучения и тестирования:")
    print(f"X_train:\n{X_train.head()}")
    print(f"y_train:\n{y_train.head()}")
    print(f"X_test:\n{X_test.head()}")
    print(f"y_test:\n{y_test.head()}")

    # Сброс индексов и преобразование данных в строковый формат
    X_train = X_train.reset_index(drop=True).astype(str)
    X_test = X_test.reset_index(drop=True).astype(str)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Создание DataFrame с текстовыми признаками
    train_df = pd.DataFrame({'text': X_train})
    test_df = pd.DataFrame({'text': X_test})

    learn_pool = Pool(data=train_df, label=y_train, text_features=['text'])
    test_pool = Pool(data=test_df, label=y_test, text_features=['text'])

    catboost_default_params = {
        'iterations': 50,
        'learning_rate': 0.03,
        'eval_metric': 'Accuracy',
    }

    catboost_default_params.update(catboost_params)

    model = CatBoostClassifier(**catboost_default_params)
    model.fit(learn_pool, eval_set=test_pool, verbose=verbose)

    return model

# Обучение моделей и сохранение
model_save_path = "models"
os.makedirs(model_save_path, exist_ok=True)

for idx, (X_train, X_test, y_train, y_test) in enumerate(dfs_split):
    if len(y_train.unique()) > 1 and len(y_test.unique()) > 1:
        model = fit_catboost(X_train, X_test, y_train, y_test)
        model.save_model(os.path.join(model_save_path, f'catboost_model_category_{idx}.cbm'))
        print(f"Model for category {idx} saved.")
    else:
        print(f"Skipped category {idx} due to lack of class diversity.")

print("All models trained and saved successfully.")
