import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# Загрузите тексты из файла
with open("texts/texts3.txt", "r", encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines() if line.strip()]


labels = np.loadtxt("tags/saved_tags.csv", delimiter=",")

# # Убедитесь, что количество текстов совпадает с количеством меток
assert len(texts) == labels.shape[0]

# Инициализируйте модель sentence-transformer
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Получите эмбеддинги для ваших текстов
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# Разделите данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Используйте логистическую регрессию для мультилейбл классификации
logreg = LogisticRegression(solver='liblinear')
multi_target_logreg = MultiOutputClassifier(logreg, n_jobs=-1)

# Обучите модель
# multi_target_logreg.fit(X_train, y_train)

# Оцените модель на тестовой выборке
# print("Точность на тестовой выборке:", multi_target_logreg.score(X_test, y_test))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение и тестирование с использованием X_train_scaled и X_test_scaled
multi_target_logreg.fit(X_train_scaled, y_train)
print("Точность на тестовой выборке:", multi_target_logreg.score(X_test_scaled, y_test))
