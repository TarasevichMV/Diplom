import numpy as np
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# Загрузите тексты из файла
with open("texts/texts3.txt", "r", encoding='utf-8') as f:
    texts = [simple_preprocess(line.strip()) for line in f.readlines() if line.strip()]

# Загрузите метки из файла
labels = np.loadtxt("tags/saved_tags.csv", delimiter=",")

# Убедитесь, что количество текстов совпадает с количеством меток
assert len(texts) == labels.shape[0]

# Создайте словарь и корпус для LDA
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Обучите модель LDA
num_topics = 10  # Вы можете изменить это число в зависимости от вашего набора данных
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# Просмотрите темы в модели LDA
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Преобразуйте распределение тем в фиксированные векторы для каждого документа
features = np.array([lda_model.get_document_topics(doc, minimum_probability=0.0) for doc in corpus])
features = np.array([[prob for _, prob in doc] for doc in features])

# Разделите данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Инициализация модели RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Инициализация MultiOutputClassifier
multi_target_rf = MultiOutputClassifier(rf, n_jobs=-1)

# Обучение модели
multi_target_rf.fit(X_train, y_train)

# Предсказание
predictions = multi_target_rf.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, predictions)
print(f"Точность на тестовой выборке: {accuracy}")
