# import autokeras as ak

# # Предположим, что labels уже закодированы должным образом
# # text_vectors - это ваши векторные представления текстов
# # labels - это ваши метки

# # Разделение данных на обучающую и тестовую выборки
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(text_vectors, labels, test_size=0.2)

# # Создание модели TextClassifier
# text_classifier = ak.TextClassifier(
#     overwrite=True,
#     max_trials=3 # Количество попыток для поиска модели
# )

# # Запуск поиска
# text_classifier.fit(x_train, y_train, epochs=2)

# # Оценка модели
# loss, accuracy = text_classifier.evaluate(x_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

# # Применение модели для предсказания
# predictions = text_classifier.predict(x_test)
