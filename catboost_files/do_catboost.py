import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Добавление пути к проекту
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Чтение тегов
tags_file_path = os.path.join(os.path.dirname(__file__), '..', 'tags', 'tags.csv')
tags_file_path = os.path.abspath(tags_file_path)
tags_df = pd.read_csv(tags_file_path, header=None)

# Чтение текстов
texts_file_path = os.path.join(os.path.dirname(__file__), '..', 'texts', 'texts.csv')
texts_file_path = os.path.abspath(texts_file_path)
texts_df = pd.read_csv(texts_file_path, encoding='utf-8', header=None)

# Проверка и удаление пустых строк
texts_df.dropna(inplace=True)
tags_df.dropna(inplace=True)

# Проверка количества строк
texts_count = texts_df.shape[0]
tags_count = tags_df.shape[0]

if texts_count != tags_count:
    print(f"Ошибка: количество текстов ({texts_count}) не совпадает с количеством меток ({tags_count}).")
    sys.exit(1)
else:
    print(f"Количество текстов и меток совпадает: {texts_count} строк.")

# Преобразование текстов и меток в массивы
texts = texts_df.iloc[:, 0].tolist()
labels = tags_df.to_numpy()

# Преобразование текстов и меток в массивы numpy
texts = np.array(texts)
labels = np.array(labels)

# Подготовка данных для модели
categories = list(range(labels.shape[1]))
dfs = [pd.DataFrame({"text": texts, "label": labels[:, i]}) for i in categories]
dfs_split = [train_test_split(df.text, df["label"].astype(float), test_size=0.2, random_state=42) for df in dfs]

# Функция для обучения модели CatBoost
def fit_catboost(X_train, X_test, y_train, y_test, catboost_params={}, verbose=100):
    learn_pool = Pool(
        X_train,
        y_train,
        text_features=["text"]
    )
    test_pool = Pool(
        X_test,
        y_test,
        text_features=["text"]
    )

    catboost_default_params = {
        'iterations': 50,
        'learning_rate': 0.03,
        'eval_metric': 'Accuracy',
    }

    catboost_default_params.update(catboost_params)

    model = CatBoostClassifier(**catboost_default_params)
    model.fit(learn_pool, eval_set=test_pool, verbose=verbose)

    return model, test_pool, model.get_evals_result()

# Обучение моделей для каждой категории и расчет метрик
overall_test_labels = []
overall_predicted_labels = []
histories = []

model_save_path = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

for idx, df_split in enumerate(dfs_split):
    model, test_pool, history = fit_catboost(pd.DataFrame(df_split[0]), pd.DataFrame(df_split[1]), pd.DataFrame(df_split[2]), pd.DataFrame(df_split[3]))
    predicted_labels = model.predict(test_pool).astype(float)
    overall_test_labels.append(df_split[3].values.astype(float))
    overall_predicted_labels.append(predicted_labels)
    histories.append(history)
    
    # Сохранение модели
    model_file = os.path.join(model_save_path, f'catboost_model_category_{idx}.cbm')
    model.save_model(model_file)
    print(f"Модель для категории {idx} сохранена в {model_file}")

overall_test_labels = np.concatenate(overall_test_labels)
overall_predicted_labels = np.concatenate(overall_predicted_labels)

# Конвертация меток в плоский вид для вычисления метрик
overall_test_labels_flat = overall_test_labels.flatten()
overall_predicted_labels_flat = overall_predicted_labels.flatten()

# Вычисление метрик
report = classification_report(overall_test_labels_flat, (overall_predicted_labels_flat > 0.5).astype(int), target_names=["0", "1"], output_dict=True, zero_division=0)
print(f"Макро-усредненная точность: {report['macro avg']['precision']}")
print(f"Макро-усредненная полнота: {report['macro avg']['recall']}")
print(f"Макро-усредненная F1-мера: {report['macro avg']['f1-score']}")

# Построение матрицы ошибок
for idx, (test_labels, predicted_labels) in enumerate(zip(overall_test_labels, overall_predicted_labels)):
    cm = confusion_matrix(test_labels, (predicted_labels > 0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot()
    plt.title(f'Матрица ошибок для категории {idx}')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()

# Графики потерь и точности
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for idx, history in enumerate(histories):
    plt.plot(history['learn']['Accuracy'], label=f'Точность на обучении {idx}')
    plt.plot(history['validation']['Accuracy'], label=f'Точность на валидации {idx}')
plt.title('Точность')
plt.legend()

plt.subplot(1, 2, 2)
for idx, history in enumerate(histories):
    plt.plot(history['learn']['Logloss'], label=f'Потери на обучении {idx}')
    plt.plot(history['validation']['Logloss'], label=f'Потери на валидации {idx}')
plt.title('Потери')
plt.legend()

plt.show()
