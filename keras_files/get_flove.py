from numpy import array
from numpy import asarray
from numpy import zeros

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
