from telethon import TelegramClient
import os
from dotenv import load_dotenv
import sqlite3
from pytz import timezone
import time

# Загрузка переменных среды
load_dotenv()
api_id = os.environ.get('api_id')
api_hash = os.environ.get('api_hash')
phone = os.environ.get('phone')

utc = timezone('Europe/Moscow')
timezone = timezone('Europe/Moscow')

# Настройки базы данных
db_path = 'mydatabase.db'

# Подключение к Telegram
client = TelegramClient('session.session', api_id, api_hash, system_version="4.16.30-vxvadUSTOM")
client.start()

existing_messages = []

# Подключение к базе данных
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Создание таблицы, если она еще не существует
cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER,
        message_date TEXT,
        message_text TEXT,
        channel_username TEXT
    )
''')
conn.commit()

# Получение существующих сообщений
query = "SELECT channel_username, message_id FROM messages"
cursor.execute(query)
rows = cursor.fetchall()
for row in rows:
    existing_messages.append((row[0], row[1]))

# Закрытие подключения к базе данных
conn.close()

# Список каналов для парсинга
companies_channels = [
    'evrazcom',
    'uralsteel_official',
    'metalloinvest_official',
    'MMK_official',
    'nlmk_official',
    'severstal',
    'tmk_group'
]

# Парсинг сообщений
for channel in companies_channels:
    print(f'Начат парсинг канала: {channel}')
    for message in client.iter_messages(channel):
        message_date = message.date.astimezone(timezone)
        if (channel, message.id) not in existing_messages and message.text:
            print(f'Добавлено сообщение в базу данных: {channel} {message.id}')
            
            # Подключение к базе данных для вставки нового сообщения
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO messages (message_id, message_date, message_text, channel_username)
                VALUES (?, ?, ?, ?)
            ''', (message.id, message_date.isoformat(), message.text, channel))
            conn.commit()
            conn.close()
            
            # Добавляем новое сообщение в существующий список, чтобы не вставлять его повторно
            existing_messages.append((channel, message.id))
        elif (channel, message.id) in existing_messages:
            print(f'Закончена парсинг канала {channel}')
            break
        time.sleep(0.1)

print('Парсинг завершен.')
client.disconnect()
