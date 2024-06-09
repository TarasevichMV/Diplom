import sqlite3

# Путь к базе данных
db_path = 'mydatabase.db'

# Подключение к базе данных
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Запрос на обновление всех значений в столбце analized на 0
update_query = "UPDATE messages SET analized = 0 WHERE analized = 1"
cursor.execute(update_query)

# Коммит изменений и закрытие подключения
conn.commit()
conn.close()

print("Все значения в столбце 'analized' были успешно обновлены на 0.")
