import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Разделение данных на тренировочную и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Подготовка данных для модели
train_pool = Pool(train_texts, label=train_labels, text_features=[0])
test_pool = Pool(test_texts, label=test_labels, text_features=[0])

# Задание сетки гиперпараметров для поиска
param_grid = {
    'iterations': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128]
}

# Настройка и обучение модели CatBoost с GridSearchCV
model = CatBoostClassifier(loss_function='MultiLogloss', eval_metric='Accuracy', verbose=0)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=10, n_jobs=-1)

grid_search.fit(train_texts, train_labels)

# Вывод лучших гиперпараметров и результатов
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

# Обучение лучшей модели на тренировочном наборе и оценка на тестовом наборе
best_model = grid_search.best_estimator_
best_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# Сохранение лучшей модели
model_save_path = os.path.join(os.path.dirname(__file__), 'best_catboost_model.cbm')
best_model.save_model(model_save_path)
print(f"Best model saved to {model_save_path}")

# Предсказание меток
predicted_labels = best_model.predict(test_pool)

# Преобразование предсказанных меток и истинных меток в формат numpy
predicted_labels = np.array(predicted_labels)
test_labels = np.array(test_labels)

# Вычисление метрик
report = classification_report(test_labels, predicted_labels, output_dict=True, zero_division=0)
print(f"Macro-averaged Precision: {report['macro avg']['precision']}")
print(f"Macro-averaged Recall: {report['macro avg']['recall']}")
print(f"Macro-averaged F1-score: {report['macro avg']['f1-score']}")

# Определение уникальных меток в тестовой выборке
unique_labels = np.unique(np.concatenate((test_labels.argmax(axis=1), predicted_labels.argmax(axis=1))))

# Построение матрицы ошибок
cm = confusion_matrix(test_labels.argmax(axis=1), predicted_labels.argmax(axis=1), labels=unique_labels)
cm_labels = [f"Class {i}" for i in unique_labels]  # Используйте реальные метки, если они доступны

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
disp.plot()
plt.show()

# Графики потерь и точности
evals_result = best_model.get_evals_result()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(evals_result['learn']['Accuracy'], label='Train Accuracy')
plt.plot(evals_result['validation']['Accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(evals_result['learn']['MultiLogloss'], label='Train Loss')
plt.plot(evals_result['validation']['MultiLogloss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()
