from analize import analize
import pandas as pd
from datetime import datetime
import os


# Список текстов для классификации
directory = 'texts'
texts = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)

results = analize(texts)

# Предполагаем, что у вас есть данные вероятностей для каждого текста и каждой модели
# и вам нужно установить пороговое значение для классификации как "1"
threshold = 0.8

# Заголовки для DataFrame
columns = ["Дата", "Инвестиции", "Производство", "Операционка", "Новые рынки/сегменты", "Социалка",
           "Сервис", "ESG", "Персонал", "Импортозамещение", "R&D", "НВП", "Ремонты", "Туризм", "Финансы",
           "Цифровизация", "PR", "Нефтегаз", "Энергетика", "Строительство", "Машиностроение", "Прочие",
           "Текст новости"]

# Создаем DataFrame
df = pd.DataFrame(columns=columns)

rows = []  # Список для хранения данных строк перед добавлением в DataFrame

for text_index, text in enumerate(texts):
    row = {column: 0 for column in columns}
    row["Дата"] = datetime.now().strftime("%Y-%m-%d")
    row["Текст новости"] = text

    # Перебор моделей и проверка вероятности превышения порога
    for model_index in range(21):
        probability = results[f"Model {model_index}"][text_index]
        if probability > threshold:
            tag_name = columns[model_index + 1]  # +1, так как первая колонка - это дата
            row[tag_name] = 1

    rows.append(row)  # Добавляем собранную строку в список

# Добавляем все собранные строки в DataFrame одним вызовом pd.concat
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

# Сохраняем DataFrame в файл Excel
df.to_excel("classified_texts1.xlsx", index=False, engine='openpyxl')
