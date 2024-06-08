from telethon import TelegramClient
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
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
engine = create_engine('sqlite:///../mydatabase.db')
Base = declarative_base()


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer)
    message_date = Column(DateTime)
    message_text = Column(String)
    channel_username = Column(String)


# Создание таблицы, если она еще не существует
Base.metadata.create_all(engine)

# Настройки сессии для ORM
Session = sessionmaker(bind=engine)

# Подключение к Telegram
client = TelegramClient('session.session', api_id, api_hash, system_version="4.16.30-vxvadUSTOM")
client.start()

# Получение списка существующих сообщений
with Session() as session:
    existing_messages = session.query(Message.channel_username, Message.message_id).all()

# Преобразуем список существующих сообщений в набор для быстрого поиска
existing_messages_set = {(msg.channel_username, msg.message_id) for msg in existing_messages}

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
        if (channel, message.id) not in existing_messages_set and message.text:
            with Session() as session:

                # Создаем и добавляем новое сообщение в базу данных
                db_message = Message(
                    message_id=message.id,
                    message_date=message_date,
                    message_text=message.text,
                    channel_username=channel
                )
                session.add(db_message)
                session.commit()
        elif (channel, message.id) in existing_messages_set:
            print(f'Закончена парсинг канала {channel}')
            break
        time.sleep(0.1)

print('Парсинг завершен.')
client.disconnect()
