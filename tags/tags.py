import csv


def get_tags():
    '''
    Метод для получения тегов из файла tags.csv
    :return: список списков, содержащий теги для каждого столбца
    tags.csv составлен из тегов файла Анализ стратегий конкурентов.xlsx
    '''
    file_path = 'tags/tags.csv'

    # Инициализация списков для каждого столбца
    columns = [[] for _ in range(21)]

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for i in range(21):
                columns[i].append(1 if row[i] == '+' else 0)

    # # Печать результатов для проверки
    # for index, column in enumerate(columns):
    #     print(f"Column {index + 1}: {column}")
    return columns
