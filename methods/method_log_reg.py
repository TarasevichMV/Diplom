import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_logistic_regression(text_vectors: np.ndarray, labels: list, test_size: float) -> str:
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(text_vectors, labels, test_size=test_size, random_state=42)
    # Обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the logistic regression model: {accuracy:.2f}")
    return model
