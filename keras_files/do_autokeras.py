import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
import autokeras as ak
import sys

# Добавление пути к проекту
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from preprocessing.textPreprocessor import fasttext_lowercase_train_config, TextPreprocessor

# Инициализация препроцессора
PREPROCESSOR = TextPreprocessor(fasttext_lowercase_train_config)

# Чтение тегов
file_path = os.path.join(os.path.dirname(__file__), '..', 'tags', 'tags.csv')
file_path = os.path.abspath(file_path)
tags_df = pd.read_csv(file_path, header=None)

# Чтение текстов
file_path = os.path.join(os.path.dirname(__file__), '..', 'texts', 'texts.csv')
file_path = os.path.abspath(file_path)
texts_df = pd.read_csv(file_path, encoding='utf-8', header=None)

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

# Создание модели с помощью AutoKeras
clf = ak.TextClassifier(overwrite=True, max_trials=1)

# Обучение модели
history = clf.fit(train_texts, train_labels, epochs=20, validation_data=(test_texts, test_labels))

# Оценка модели на тестовой выборке
evaluation = clf.evaluate(test_texts, test_labels)
print(f"Test Accuracy: {evaluation[1]}")
print(f"Test Loss: {evaluation[0]}")

# Прогнозирование меток на тестовых данных
predicted_labels = clf.predict(test_texts)

# Преобразование предсказанных меток в формат numpy
predicted_labels = np.array(predicted_labels).squeeze()

# Вычисление метрик
columns = ["Инвестиции", "Производство", "Операционка", "Новые рынки/сегменты",
           "Социалка", "Сервис", "ESG", "Персонал", "Импортозамещение", "R&D",
           "НВП", "Ремонты", "Туризм", "Финансы", "Цифровизация", "PR",
           "Нефтегаз", "Энергетика", "Строительство", "Машиностроение",
           "Прочие"]
report = classification_report(test_labels, (predicted_labels > 0.5).astype(int), target_names=columns, output_dict=True, zero_division=0)
print(f"Macro-averaged Precision: {report['macro avg']['precision']}")
print(f"Macro-averaged Recall: {report['macro avg']['recall']}")
print(f"Macro-averaged F1-score: {report['macro avg']['f1-score']}")

# Корректировка количества меток для матрицы ошибок
unique_labels = np.unique(np.concatenate((test_labels.argmax(axis=1), predicted_labels.argmax(axis=1))))
cm_labels = [columns[i] for i in unique_labels]

cm = confusion_matrix(test_labels.argmax(axis=1), predicted_labels.argmax(axis=1), labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
disp.plot()
plt.show()

# Графики потерь и точности
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.show()
