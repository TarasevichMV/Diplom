import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys
import pickle

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

# Преобразование текстов в векторы
vectorizer = TextVectorization(max_tokens=20000, output_mode='int', output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(train_texts).batch(128)
vectorizer.adapt(text_ds)

train_texts_vectorized = vectorizer(train_texts)
test_texts_vectorized = vectorizer(test_texts)

# Создание модели
model = Sequential([
    layers.Embedding(input_dim=20001, output_dim=128, input_length=200),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(labels.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(train_texts_vectorized, train_labels, epochs=20, validation_data=(test_texts_vectorized, test_labels), callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# Создание директории для сохранения модели
model_save_dir = 'saved_model'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'my_model.keras')
vectorizer_path = os.path.join(model_save_dir, 'vectorizer.pkl')

# Сохранение модели
model.save(model_save_path)
print(f"Модель сохранена в '{model_save_path}'")

# Сохранение vectorizer
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"Vectorizer сохранен в '{vectorizer_path}'")

# Оценка модели на тестовой выборке
evaluation = model.evaluate(test_texts_vectorized, test_labels)
print(f"Test Accuracy: {evaluation[1]}")
print(f"Test Loss: {evaluation[0]}")

# Прогнозирование меток на тестовых данных
predicted_labels = model.predict(test_texts_vectorized)

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
