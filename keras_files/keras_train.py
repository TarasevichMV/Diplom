from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

import csv
from classes import fasttext_lowercase_train_config, TextPreprocessor
from tags2 import get_tags

from numpy import asarray
from numpy import zeros


TEST_SIZE = (0.1, 0.2, 0.3)  # размер тестовой выборки для обучения моделей
PREPROCESSOR = TextPreprocessor(fasttext_lowercase_train_config)

columns = ["Инвестиции", "Производство", "Операционка", "Новые рынки/сегменты", "Социалка",
           "Сервис", "ESG", "Персонал", "Импортозамещение", "R&D", "НВП", "Ремонты", "Туризм", "Финансы",
           "Цифровизация", "PR", "Нефтегаз", "Энергетика", "Строительство", "Машиностроение", "Прочие",
           "Текст новости"]

texts = []  # список для хранения текстов
labels = np.array(get_tags())  # список для хранения тегов

# Чтение текстов
# with open('texts.csv', errors='replace', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         string = row[0]
#         string = PREPROCESSOR.text_cleaning(string)
#         texts.append(string)

# Запись текстов в файл
# with open('texts3.txt', 'w', encoding='utf-8') as f:
#     for item in texts:
#         f.write("%s\n" % item)

# Чтение текстов из файла
with open('texts3.txt', 'r', encoding='utf-8') as f:
    for line in f:
        texts.append(line)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# 12.05.2024
# Словарь для хранения векторов
embeddings_dictionary = dict()

# Путь к файлу изменен на ваш путь к скачанной модели
model_file = open('models/rus_model.txt', encoding="utf8")

for line in model_file:
    records = line.split()
    word = records[0]
    # Преобразование строк векторных значений в массив numpy
    # Указывайте здесь правильную размерность, например, 300
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

model_file.close()

# Инициализация матрицы вложения
# Замените 100 на размерность вашей модели, например, 300
embedding_matrix = zeros((vocab_size, 300))

for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        # Если представление слова найдено, добавляем его в матрицу
        embedding_matrix[index] = embedding_vector

print(embedding_matrix)


deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(21, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())


# from keras.utils import plot_model
# plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])



import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
