import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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

# Загрузка BERT токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Функция токенизации
def tokenize_texts(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='tf')

train_encodings = tokenize_texts(train_texts)
test_encodings = tokenize_texts(test_texts)

# Определение модели
input_ids = Input(shape=(train_encodings['input_ids'].shape[1],), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(train_encodings['attention_mask'].shape[1],), dtype=tf.int32, name='attention_mask')
bert_output = bert_model([input_ids, attention_mask])[1]
dropout = Dropout(0.3)(bert_output)
output = Dense(labels.shape[1], activation='sigmoid')(dropout)

model = Model(inputs=[input_ids, attention_mask], outputs=output)

# Компиляция модели
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Вычисление весов классов
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels.argmax(axis=1)), y=labels.argmax(axis=1))
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Обучение модели
history = model.fit(
    [train_encodings['input_ids'], train_encodings['attention_mask']],
    train_labels,
    validation_data=([test_encodings['input_ids'], test_encodings['attention_mask']], test_labels),
    epochs=3,  # BERT требует меньше эпох для обучения
    batch_size=16,
    class_weight=class_weights
)

# Оценка модели на тестовой выборке
evaluation = model.evaluate([test_encodings['input_ids'], test_encodings['attention_mask']], test_labels)
print(f"Test Accuracy: {evaluation[1]}")
print(f"Test Loss: {evaluation[0]}")

# Прогнозирование меток на тестовых данных
predicted_labels = model.predict([test_encodings['input_ids'], test_encodings['attention_mask']])

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
