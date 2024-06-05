import pandas as pd
from hard_code import hard_code
from hard_code import hard_code

# Шаг 1: Загрузить таблицу из файла Excel с заголовками
df = pd.read_excel('news_data.xlsx')

# Шаг 2: Выбрать необходимый столбец по имени
text_column = df['Содержимое']


# Шаг 3: Создать новый столбец, применив к каждому тексту функцию
def your_function(text):
    return hard_code(text)


df['Инвестиции'] = text_column.apply(your_function)

# Шаг 4: Сохранить результирующую таблицу в новый файл Excel
df.to_excel('resulting_excel2.xlsx', index=False)

# Показать первые несколько строк результирующей таблицы
print(df.head())
