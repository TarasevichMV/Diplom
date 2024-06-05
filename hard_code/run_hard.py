import pandas as pd
from hard_code import hard_code

# Шаг 1: Загрузить исходную таблицу с текстами без заголовков
df_texts = pd.read_csv('texts/texts.csv', header=None)

# Шаг 2: Загрузить вторую таблицу и выбрать первый столбец без заголовков
df_additional = pd.read_csv('tags/saved_tags.csv', header=None)
selected_column = df_additional.iloc[:, 0]  # Выбор первого столбца

# Шаг 3: Объединить две таблицы
# Предполагаем, что обе таблицы имеют одинаковое количество строк и порядок строк соответствует
df_texts['Исходная классификация'] = selected_column


# Шаг 4: Создать третий столбец и заполнить его результатами функции
def your_function(text):
    return hard_code(text)


# Предполагаем, что тексты находятся в первом столбце (с индексом 0)
df_texts['Предлагаемая классификация'] = df_texts.iloc[:, 0].apply(your_function)

# Сохранить результирующую таблицу в новый файл .xlsx
df_texts.to_excel('hard_code/resulting_excel.xlsx', index=False)

# Показать первые несколько строк результирующей таблицы
print(df_texts.head())
