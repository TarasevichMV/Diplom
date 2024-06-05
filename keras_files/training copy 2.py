import csv
from classes import fasttext_lowercase_train_config, TextPreprocessor
from tags2 import get_tags
from sklearn.model_selection import train_test_split
import autokeras as ak
import numpy as np


# pip install scikit-learn


TEST_SIZE = (0.1, 0.2, 0.3)  # размер тестовой выборки для обучения моделей

PREPROCESSOR = TextPreprocessor(fasttext_lowercase_train_config)
texts = []  # список для хранения текстов
labels = get_tags()
columns = ["Инвестиции", "Производство", "Операционка", "Новые рынки/сегменты", "Социалка",
           "Сервис", "ESG", "Персонал", "Импортозамещение", "R&D", "НВП", "Ремонты", "Туризм", "Финансы",
           "Цифровизация", "PR", "Нефтегаз", "Энергетика", "Строительство", "Машиностроение", "Прочие",
           "Текст новости"]

# print(len(tags[0]))
# Чтение текстов
with open('texts.csv', errors='replace', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        string = row[0]
        string = PREPROCESSOR.text_cleaning(string)
        texts.append(string)

with open('texts2.txt', 'w', encoding='utf-8') as f:
    for item in texts:
        f.write("%s\n" % item)

x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

model = ak.MultiLabelClassifier(max_trials=3)
