import csv
from preprocessing.textPreprocessor import fasttext_lowercase_train_config, TextPreprocessor
from tags import get_tags
from joblib import dump
from razdel import sentenize, tokenize
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import fasttext



TEST_SIZE = (0.1, 0.2, 0.3)  # размер тестовой выборки для обучения моделей

PREPROCESSOR = TextPreprocessor(fasttext_lowercase_train_config)
dataset = []  # список для хранения текстов
tags = get_tags()
columns = ["Инвестиции", "Производство", "Операционка", "Новые рынки/сегменты", "Социалка",
           "Сервис", "ESG", "Персонал", "Импортозамещение", "R&D", "НВП", "Ремонты", "Туризм", "Финансы",
           "Цифровизация", "PR", "Нефтегаз", "Энергетика", "Строительство", "Машиностроение", "Прочие",
           "Текст новости"]

# print(len(tags[0]))
# Чтение текстов
with open('texts.csv', errors='replace', encoding='utf-8') as file:
    reader = csv.reader(file)
    j = 0
    for row in reader:
        string = row[0]
        string = PREPROCESSOR.text_cleaning(string)
        for i in range(len(tags)):
            if tags[i][j] == 1:
                string = f'__label__{columns[i]} {string}'
        dataset.append(string)
        j += 1

with open('texts.txt', 'w', encoding='utf-8') as f:
    for item in dataset:
        f.write("%s\n" % item)


# Разделяем данные на обучающую и тестовую выборку
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Запишем обучающие данные в файл
with open('texts_train.txt', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write("%s\n" % item)

# Запишем тестовые данные в файл
with open('texts_test.txt', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write("%s\n" % item)

model = fasttext.train_supervised(input='train.txt')
model.save_model('model.bin')
