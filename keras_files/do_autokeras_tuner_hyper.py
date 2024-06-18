import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras_tuner import HyperParameters, Hyperband
from tensorflow.keras.callbacks import CSVLogger
import sys

# Добавление пути к проекту
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from preprocessing.textPreprocessor import fasttext_lowercase_train_config, TextPreprocessor

# Инициализация препроцессора
PREPROCESSOR = TextPreprocessor(fasttext_lowercase_train_config)

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

# Установка гиперпараметров
def build_model(hp):
    max_tokens = hp.Int('max_tokens', min_value=5000, max_value=20000, step=5000)
    embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    epochs = hp.Int('epochs', min_value=10, max_value=100, step=10)  # Добавляем гиперпараметр epochs

    vectorize_layer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=500)
    vectorize_layer.adapt(train_texts)

    model = Sequential([
        vectorize_layer,
        Embedding(input_dim=max_tokens, output_dim=embedding_dim),
        GlobalAveragePooling1D(),
        Dropout(rate=dropout_rate),
        Dense(units=labels.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Поиск лучших гиперпараметров
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='text_classification',
    overwrite=True  # добавлено чтобы избежать перезагрузки существующего проекта
)

tuner.search_space_summary()

# Добавление колбэка для записи метрик в файл
csv_logger = CSVLogger('training_log.csv', append=True)

# Запуск поиска
tuner.search(train_texts, train_labels, epochs=50, validation_data=(test_texts, test_labels), batch_size=32, callbacks=[csv_logger])

# Получение лучших гиперпараметров и модели
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)

# Печать всех лучших гиперпараметров
print(f"Лучшее значение max_tokens: {best_hp.get('max_tokens')}")
print(f"Лучшее значение embedding_dim: {best_hp.get('embedding_dim')}")
print(f"Лучшее значение dropout_rate: {best_hp.get('dropout_rate')}")
print(f"Лучшее количество эпох: {best_hp.get('epochs')}")

# Обучение лучшей модели
history = model.fit(train_texts, train_labels, epochs=best_hp.get('epochs'), validation_data=(test_texts, test_labels), batch_size=32)

# Оценка модели на тестовой выборке
evaluation = model.evaluate(test_texts, test_labels, batch_size=32)
print(f"Точность на тестовой выборке: {evaluation[1]}")
print(f"Потери на тестовой выборке: {evaluation[0]}")

# Прогнозирование меток на тестовых данных
predicted_labels = model.predict(test_texts)

# Преобразование предсказанных меток в формат numpy
predicted_labels = np.array(predicted_labels).squeeze()

# Вычисление метрик
report = classification_report(test_labels, (predicted_labels > 0.5).astype(int), output_dict=True, zero_division=0)
print(f"Макро-усредненная точность: {report['macro avg']['precision']}")
print(f"Макро-усредненная полнота: {report['macro avg']['recall']}")
print(f"Макро-усредненная F1-меря: {report['macro avg']['f1-score']}")

# Сохранение модели
model.save('best_text_classification_model.h5')
