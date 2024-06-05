import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


def train_adaboost(text_vectors: np.ndarray, labels: list, test_size: float) -> str:
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(text_vectors, labels, test_size=test_size, random_state=42)
    # Обучение модели AdaBoost с DecisionTreeClassifier в качестве базового оценщика
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the AdaBoost model: {accuracy:.2f}")
    return model
