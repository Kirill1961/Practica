import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# TODO Извлекает dataset из openML по имени или id
#  Атрибуты:  ['data', 'target', 'frame', 'categories', 'details']
#  Explane: data.data → X (признаки), data.target → y, data.frame → единый DataFrame (X + y)
#  Если as_frame=False то data.frame → ❌ отсутствует

X = fetch_openml(
    name="adult",
    version=2,
    as_frame=True,
    return_X_y=False
)

# Извлекаем df + target -> X.frame
X = X.frame

# Пропуски NA
X_na = X.isna().any()
print('Пропуски в df : \n', X_na[X_na == True].index, '\n')

# Удаление пропусков NA
X = X.dropna()

# Проверка Удалённых пропусков
print('Пропуски в df : \n', X.isna().any()[X.isna().any() == True], '\n')

# Разделение df и target
y = X['class']
X = X.drop('class', axis=1)
print(X.shape, y.shape, '\n')

# target в бинарный вид
print(y.unique(), '\n')
y = (y == '<=50K').astype(int)
print(y.head(), '\n')


cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print('Качественные Признаки : \n', cat_features, '\n')


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="Accuracy",

    # ключевое для нашей темы
    cat_features=cat_features,

    # чтобы было видно обучение
    verbose=50,

    # фиксируем random seed (перестановки!)
    random_seed=42
)

# eval_set — валидация во время обучения
# Используется для: мониторинга метрик, early stopping, Не участвует в обучении деревьев

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test)
)

y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)

importances = pd.Series(
    model.get_feature_importance(),
    index=X.columns
).sort_values(ascending=False)

importances.head(10)

