import csv
from preprocessing.textPreprocessor import fasttext_lowercase_train_config, TextPreprocessor
from gensim.models import FastText
import multiprocessing as mpr
import os
import re
import string
from copy import copy
from typing import Iterable, Union, List
from tags import get_tags
from joblib import dump

import pymorphy2
from gensim.models import FastText
from nltk.corpus import stopwords
from razdel import sentenize, tokenize
from tqdm import tqdm
import numpy as np


TEST_SIZE = (0.1, 0.2, 0.3)  # размер тестовой выборки для обучения моделей

PREPROCESSOR = TextPreprocessor(fasttext_lowercase_train_config)
dataset = []  # список для хранения текстов

# Чтение текстов
with open('texts.csv', errors='replace', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row[0])

# Получаем теги
labels = get_tags()

sentences = []  # список для хранения предложений

'''
Создаем список sentences,
содержащий все предложения из dataset,
преобразованные в список токенов.
'''
for article in tqdm(dataset, desc=f"{'Getting sentences'}"):
    sentences.extend(
        [[x.text for x in tokenize(PREPROCESSOR.text_cleaning(sentence.text))] for sentence in sentenize(article)]
    )

# Определяем параметры модели
model = fasttext_lowercase_train_config["model"](**fasttext_lowercase_train_config["model_params"])

# Строим словарь
model.build_vocab(corpus_iterable=sentences)

# Учим модель. Проще учить по эпохам
for epoch in range(fasttext_lowercase_train_config.get("epochs")):
    model.train(
        corpus_iterable=tqdm(sentences, desc="Training model"),
        total_examples=len(sentences),
        epochs=1,
    )

model.save("models/fasttext_model1.bin")

# словарь, который у вас соберется в процессе
vocab = model.wv.key_to_index

'''
Обработка текста на уровне статей (article), а не на уровне предложений.
Каждая статья разбивается на предложения, а затем усредняются векторы предложений, 
чтобы получить вектор статьи.
'''
text_vectors = []
for article in dataset:
    article_sentences = list(sentenize(article))
    article_vectors = []
    for sentence in article_sentences:
        word_vectors = [model.wv[word.text] for word in tokenize(sentence.text) if word.text in model.wv]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(vector_size)  # используем вектор нулей, если список слов пуст
        article_vectors.append(sentence_vector)
    if article_vectors:
        article_vector = np.mean(article_vectors, axis=0)
    else:
        article_vector = np.zeros(vector_size)  # используем вектор нулей, если вся статья не содержит известных слов
    text_vectors.append(article_vector)
text_vectors = np.array(text_vectors)

for j in TEST_SIZE:
    print(f"Test size: {j}")

    # Метод логистической регрессии
    from methods.method_log_reg import train_logistic_regression
    for i in range(21):
        clf_log_reg = train_logistic_regression(text_vectors, labels[i], j)
        dump(clf_log_reg, f"models/log_reg_model_{i}_{j}.joblib")
        print(clf_log_reg)

    # Метод наивного Байеса
    from methods.method_bayes import train_naive_bayes
    for i in range(21):
        clf_bayes = train_naive_bayes(text_vectors, labels[i], j)
        dump(clf_bayes, f"models/bayes_model_{i}_{j}.joblib")
        print(clf_bayes)

    # Метод опорных векторов
    from methods.method_svm import train_svm
    for i in range(21):
        clf_svm = train_svm(text_vectors, labels[i], j)
        dump(clf_svm, f"models/svm_model_{i}_{j}.joblib")
        print(clf_svm)

    # Метод градиентного бустинга
    from methods.method_gradient_boosting import train_gradient_boosting
    for i in range(21):
        clf_gb = train_gradient_boosting(text_vectors, labels[i], j)
        dump(clf_gb, f"models/gb_model_{i}_{j}.joblib")
        print(clf_gb)
