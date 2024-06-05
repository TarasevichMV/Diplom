import csv


def save_tags_to_csv(tags, file_path='saved_tags.csv'):
    '''
    Метод для сохранения тегов в CSV файл без заголовков.
    :param tags: список списков, содержащий теги для каждого столбца.
    :param file_path: путь для сохранения файла CSV.
    '''
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        for tag_row in tags:
            writer.writerow(tag_row)


def transform_row(row):
    # Разделение строки по запятым и преобразование в бинарный список
    return [1 if x == '+' else 0 for x in row.split(',')]


def get_tags(file_path='tags.csv'):
    '''
    Метод для получения тегов из файла tags.csv
    :return: список списков, содержащий теги для каждого столбца
    tags.csv составлен из тегов файла Анализ стратегий конкурентов.xlsx
    '''
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # Применение transform_row к каждой строке
    tags = [transform_row(line) for line in lines]

    return tags
