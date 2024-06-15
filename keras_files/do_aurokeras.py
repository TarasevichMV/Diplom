import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from tensorflow.keras.callbacks import CSVLogger, Callback
import autokeras as ak
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

class MetricsCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.precision = []
        self.recall = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs=None):
        val_texts, val_labels = self.validation_data
        val_predictions = self.model.predict(val_texts)
        val_predictions = (val_predictions > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='macro', zero_division=0)
        
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)

# Создание AutoKeras модели
input_node = ak.TextInput()
output_node = ak.TextToIntSequence(max_tokens=20000, output_sequence_length=500)(input_node)
output_node = ak.Embedding()(output_node)
output_node = ak.DenseBlock()(output_node)
output_node = ak.ClassificationHead(num_classes=labels.shape[1])(output_node)

auto_model = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    max_trials=1,  # Установка в 1, так как мы не делаем тюнинг гиперпараметров
    directory='my_dir',
    project_name='text_classification',
    overwrite=True
)

# Обучение модели
csv_logger = CSVLogger('training_log.csv', append=True)
metrics_callback = MetricsCallback(validation_data=(test_texts, test_labels))
history = auto_model.fit(train_texts, train_labels, epochs=50, validation_data=(test_texts, test_labels), batch_size=32, callbacks=[csv_logger, metrics_callback])

# Оценка модели на тестовой выборке
evaluation = auto_model.evaluate(test_texts, test_labels, batch_size=32)
print(f"Точность на тестовой выборке: {evaluation[1]}")
print(f"Потери на тестовой выборке: {evaluation[0]}")

# Прогнозирование меток на тестовых данных
predicted_labels = auto_model.predict(test_texts)

# Преобразование предсказанных меток в формат numpy
predicted_labels = np.array(predicted_labels).squeeze()

# Вычисление метрик
columns = ["Инвестиции", "Производство", "Операционка", "Новые рынки/сегменты",
           "Социалка", "Сервис", "ESG", "Персонал", "Импортозамещение", "R&D",
           "НВП", "Ремонты", "Туризм", "Финансы", "Цифровизация", "PR",
           "Нефтегаз", "Энергетика", "Строительство", "Машиностроение",
           "Прочие"]
report = classification_report(test_labels, (predicted_labels > 0.5).astype(int), target_names=columns, output_dict=True, zero_division=0)
print(f"Макро-усредненная точность: {report['macro avg']['precision']}")
print(f"Макро-усредненная полнота: {report['macro avg']['recall']}")
print(f"Макро-усредненная F1-меря: {report['macro avg']['f1-score']}")

# Корректировка количества меток для матрицы ошибок
unique_labels = np.unique(np.concatenate((test_labels.argmax(axis=1), predicted_labels.argmax(axis=1))))
cm_labels = [str(i) for i in unique_labels]

cm = confusion_matrix(test_labels.argmax(axis=1), predicted_labels.argmax(axis=1), labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
disp.plot()
plt.title('Матрица ошибок')
plt.show()

# Графики метрик
epochs = range(1, len(metrics_callback.precision) + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, metrics_callback.precision, label='Macro-averaged Precision')
plt.title('Macro-averaged Precision')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, metrics_callback.recall, label='Macro-averaged Recall')
plt.title('Macro-averaged Recall')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, metrics_callback.f1, label='Macro-averaged F1-score')
plt.title('Macro-averaged F1-score')
plt.legend()

plt.show()

# Сохранение модели
auto_model.export_model().save('best_text_classification_model', save_format='tf')
