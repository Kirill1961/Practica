"""
Сглаживание (на основе смешения),
Кастомное CV,
Кастомное заполнения:
    * заполняем значением:
    fill = (значения из окна[-100:] - прошлое локальное mean (окна [-100:])) + локальное mean (окна [-20:])
Удаление аутлаеров и мусора:
    * Отбор аутлаеров:
    mean = значения из окна [-500:] - прошлое локальное mean (окна [-500:])
    std = значения из окна [-500:] - прошлое локальное std (окна [-500:])
    outlier = mean + std * sigma
A_rate → расход газа в точке A
B_rate → расход газа в точке B

size=9792
"""

# TODO START
# импорт библиотек
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import Ridge
from scipy.signal import find_peaks, peak_prominences
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-bright')

from etna.analysis import (
    acf_plot,
    plot_correlation_matrix,
    cross_corr_plot,
    distribution_plot,
    plot_trend,
    stl_plot,
    seasonal_plot,
    plot_periodogram,
    get_anomalies_density,
    get_anomalies_median,
    plot_anomalies,
    find_change_points,
    plot_change_points_interactive,
    plot_time_series_with_change_points,
    plot_forecast
)
from etna.datasets.tsdataset import TSDataset
from etna.transforms import LinearTrendTransform, LagTransform, TimeSeriesImputerTransform
from etna.models import NaiveModel, LinearPerSegmentModel, CatBoostPerSegmentModel, SeasonalMovingAverageModel
from etna.metrics import MAE, MAPE
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform


#%%


# TODO 1 часть: функции
def mean_abs_per_err(y_true, y_pred):
    """
    Функция подсчета MAPE
    """
    return (abs((y_true - y_pred) / y_true)).mean() * 100


def exponential_smoothing(series, alpha):
    """
    Функция экспоненциального сглаживания, Рекурсивно.
    Получаем на вход сериес, и последовательно сглаживаем значения с "силой" alpha и сдвигом 1 шаг в перёд
    """
    result = [series[0]]

    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])

    return pd.Series(result)


# TODO Кастомный ffill
def A_B_rate_restore(a_rate, b_rate, window, sigma):
    """
    Кастомный ffill()
    Функция, которая восстанавливает значения A_rate и B_rate
    В окне считаем статистики, затем основываясь на среднеквадратичном отклонении определяем аутлаеры
    Если после этого у нас не хватает только одного значения из пары A_rate и B_rate, то строим регрессию
    И восстанавливаем второе. Если неизвестны оба, заполняем на основе предыдущих значений
    Руками находим значения для окон 20 и 100 через CV
    Размер окон 20 и 100, при infer_freq = 30min, 20 - 10 часов и 100 - 50часов примерно 2 дня,
    * 10 часов локальной динамики
    * 2 суток устойчивого направления
    """
    a_rate = a_rate.copy()
    b_rate = b_rate.copy()

    result_a = [a_rate[0]]
    result_b = [b_rate[0]]

    for n in range(1, len(a_rate)):
        mean_a_gup = np.array(result_a[-20:]).mean()
        resid_a_gup = (pd.Series(result_a[-100:]) - pd.Series(result_a[-100:]).shift(1)).mean()
        mean_a_gup += resid_a_gup
        resid_mean_a = abs(pd.Series(result_a[-window:]) - pd.Series(result_a[-window:]).shift(1)).mean()
        resid_std_a = abs(pd.Series(result_a[-window:]) - pd.Series(result_a[-window:]).shift(1)).std()
        mean_b_gup = np.array(result_b[-20:]).mean()
        resid_b_gup = (pd.Series(result_b[-100:]) - pd.Series(result_b[-100:]).shift(1)).mean()
        mean_b_gup += resid_b_gup

        resid_mean_b = abs(pd.Series(result_b[-window:]) - pd.Series(result_b[-window:]).shift(1)).mean()
        resid_std_b = abs(pd.Series(result_b[-window:]) - pd.Series(result_b[-window:]).shift(1)).std()

        if n < window:
            pass
        else:
            if abs(a_rate[n] - result_a[n - 1]) > resid_mean_a + resid_std_a * sigma:
                a_rate[n] = None

            if abs(b_rate[n] - result_b[n - 1]) > resid_mean_b + resid_std_b * sigma:
                b_rate[n] = None

        if pd.isnull(a_rate[n]) and pd.isnull(b_rate[n]):
            result_a.append(mean_a_gup)
            result_b.append(mean_b_gup)

        elif pd.isnull(a_rate[n]):

            model = Ridge()
            model.fit(np.array(result_b[-3000:]).reshape(-1, 1), result_a[-3000:])
            result_a.append(model.predict(np.array(b_rate[n]).reshape(-1, 1))[0])

            result_b.append(b_rate[n])


        elif pd.isnull(b_rate[n]):

            model = Ridge()
            model.fit(np.array(result_a[-3000:]).reshape(-1, 1), result_b[-3000:])
            result_b.append(model.predict(np.array(a_rate[n]).reshape(-1, 1))[0])

            result_a.append(a_rate[n])


        else:
            result_a.append(a_rate[n])
            result_b.append(b_rate[n])

    return pd.Series(result_a), pd.Series(result_b)


def chemical_data_restore(series, window, window_mean, window_resid, sigma):
    """
    Функция, которая восстанавливает значения химических элементов
    В окне считаем статистики, затем основываясь на среднеквадратичном отклонении (волатильности) определяем аутлаеры
    И на основе статистик заполняем пропуски
    """
    series = series.copy()

    result = [series[0]]

    for n in range(1, len(series)):

        mean_gup = np.array(result[-window_mean:]).mean()

        resid_gup = (pd.Series(result[-window_resid:]) - pd.Series(result[-window_resid:]).shift(1)).mean()

        mean_gup += resid_gup

        resid_mean = abs(pd.Series(result[-window:]) - pd.Series(result[-window:]).shift(1)).mean()
        resid_std = abs(pd.Series(result[-window:]) - pd.Series(result[-window:]).shift(1)).std()

        if n < window:
            pass
        else:

            if abs(series[n] - result[n - 1]) > resid_mean + resid_std * sigma:
                series[n] = None

        if pd.isnull(series[n]):
            result.append(mean_gup)
        else:
            result.append(series[n])

    return pd.Series(result)


#%%
# TODO Восстановления процентов
def restore_percent(data):
    """
    Функция восстановления процентов состава
    После восстановления пропусков некоторые суммы улетели за логичные значения
        * 👉 каждая строка - это состав смеси, признаки это компоненты, значения - % в составе
        * 👉 сумма строки д. быть в пределах 99 - 100
    Больше 100, или меньше 99,2
    Пропорционально восстанавливаем их

    """
    data = data.copy()

    data['sum'] = data.iloc[:, 1:-1].T.sum()

    data['coef'] = 99.95 / data['sum']

    outlier_indexes = data[(data['sum'] > 99.99) | (data['sum'] < 99.92)].index

    for feat in range(1, 9):
        data.iloc[outlier_indexes, feat] = data.iloc[outlier_indexes, feat] * data.iloc[outlier_indexes, -1]

    return data.iloc[:, :-2]


#%%

# TODO 2 часть: подготовка данных с новыми адаптивными сдвигами

raw_train = pd.read_csv(r"D:\Eduson_data\sb_train_features.csv")
raw_test = pd.read_csv("D:\Eduson_data\sb_test_features.csv")
raw_targets = pd.read_csv("D:\Eduson_data\sb_train_targets.csv")
# sample = pd.read_csv("D:\Eduson_data\sb_sample_submission.csv")

raw_train.drop('timestamp', axis=1, inplace=True)
raw_test.drop('timestamp', axis=1, inplace=True)
# raw_targets.drop('timestamp', axis=1, inplace=True)


data = pd.concat([raw_train, raw_targets], axis=1)
data = pd.concat([data, raw_test], axis=0).reset_index(drop=True)

data = data[2200:].reset_index(drop=True)

data = pd.concat([data, raw_test], axis=0).reset_index(drop=True)

# TODO features = raw_train + raw_targets + raw_test
features = data.iloc[:, :10]
targets = data.iloc[:, 10:]

# TODO features
start = time.time()

features['A_rate'], features['B_rate'] = A_B_rate_restore(features['A_rate'], features['B_rate'], 500, 10)

features['A_CH4'] = chemical_data_restore(features['A_CH4'], 500, 20, 100, 9)
features['A_C2H6'] = chemical_data_restore(features['A_C2H6'], 400, 20, 100, 10)
features['A_C3H8'] = chemical_data_restore(features['A_C3H8'], 500, 20, 100, 14)
features['A_iC4H10'] = chemical_data_restore(features['A_iC4H10'], 500, 20, 100, 11)
features['A_nC4H10'] = chemical_data_restore(features['A_nC4H10'], 500, 20, 100, 11)
features['A_iC5H12'] = chemical_data_restore(features['A_iC5H12'], 400, 20, 100, 8)
features['A_nC5H12'] = chemical_data_restore(features['A_nC5H12'], 400, 20, 100, 9)
features['A_C6H14'] = chemical_data_restore(features['A_C6H14'], 500, 20, 100, 18)

end = time.time()
print(end - start)

features['B_rate_roll'] = features['B_rate'].rolling(190, min_periods=1).mean()

"""
Дальше происходит процесс адаптивных лагов признаков относительно таргета, так как была найдена
взаимосвязь между напором в трубе и временем через которое газ приходит в точку назначения. 
Обычный сдвиг на константу давал результат хуже, 👉 поэтому сдвиг будет адаптивный 
Поэтому в зависимости от расхода в точке В, мы берем более или менее старые данные из прошлого.
👉 A_rate → расход газа в точке A, B_rate → расход газа в точке B
"""

new_features = features.copy()
new_features.iloc[:, :] = 0

new_targets = targets.copy()
new_targets.iloc[:, :] = 0

for i in range(features.shape[0]):
    if i < 250:
        new_features.iloc[i] = features.iloc[0]
        new_targets.iloc[i] = targets.iloc[0]
    else:

        b_rate = features.iloc[i]['B_rate_roll']

        shift = 67 / b_rate * 198

        new_features.iloc[i] = features.iloc[int(round(i - shift))]

        new_targets.iloc[i] = targets.iloc[i]

new_features = new_features.iloc[:, :10]
new_features = new_features.reset_index(drop=True)
new_targets = new_targets.reset_index(drop=True)

new_features = new_features[250:].reset_index(drop=True)
new_targets = new_targets[250:].reset_index(drop=True)

new_features = new_features.drop(range(2250, 2450), axis=0).reset_index(drop=True)
new_targets = new_targets.drop(range(2250, 2450), axis=0).reset_index(drop=True)

new_features = new_features.drop(range(1290, 1440), axis=0).reset_index(drop=True)
new_targets = new_targets.drop(range(1290, 1440), axis=0).reset_index(drop=True)

# test = new_features[-3984:].reset_index(drop=True)

data = pd.concat([new_features, new_targets], axis=1).dropna(
    subset=['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10']).reset_index(drop=True)

new_features = data.iloc[:, :11].reset_index(drop=True)
new_targets = data.iloc[:, 11:].reset_index(drop=True)

train = new_features.copy()
# train_targets = new_targets.copy()


# raw_train = pd.read_csv("D:\Eduson_data\sb_train_features.csv")
# raw_test = pd.read_csv("D:\Eduson_data\sb_test_features.csv")
# raw_targets = pd.read_csv("D:\Eduson_data\sb_train_targets.csv")
#
# raw_train.drop('timestamp', axis=1, inplace=True)
# raw_test.drop('timestamp', axis=1, inplace=True)
# # raw_targets.drop('timestamp', axis=1, inplace=True)
#
# shift = 192
# full_size = 9792
# size_test = 3984 + shift
# size_train = full_size - size_test
#
# data = pd.concat([raw_train, raw_targets], axis=1)
# data = pd.concat([data, raw_test], axis=0)
#
# data = pd.concat([data.iloc[:, :9].reset_index(drop=True),
#                   data.iloc[1:, 9].reset_index(drop=True),
#                   data.iloc[shift:, 10:].reset_index(drop=True)], axis=1)
#
# raw_train = data.iloc[:size_train, :10].reset_index(drop=True)
# raw_targets = data.iloc[:size_train, 10:].reset_index(drop=True)
# raw_test = data.iloc[size_train:-shift, :10].reset_index(drop=True)


data = pd.concat([raw_train, raw_targets], axis=1)

trash_indexes = list()

trash_indexes += range(0, 190)

trash_indexes += range(1191, 1886)

trash_indexes += range(4523, 4689)

data = data.drop(trash_indexes, axis=0).reset_index(drop=True)

raw_train = data.iloc[:, :10]
raw_targets = data.iloc[:, 10:]

trash_indexes = list()

trash_indexes += raw_train[raw_train['A_C3H8'].isnull()].index.to_list()

trash_indexes += raw_targets[raw_targets.iloc[:, 1:5].isnull().T.sum() > 0].index.to_list()

trash_indexes += range(1131, 1138)
trash_indexes += range(2822, 2828)
trash_indexes += [4032]
trash_indexes += range(4150, 4154)
trash_indexes += range(4214, 4217)
trash_indexes += range(4345, 4348)

trash_indexes += range(2110, 2112)
trash_indexes += range(4492, 4495)

trash_indexes += [3850]
trash_indexes += range(843, 871)
trash_indexes += range(1250, 1254)

trash_indexes += range(3850, 3855)
trash_indexes += range(2829, 2831)
trash_indexes += range(4348, 4350)
trash_indexes += range(1475, 1478)

trash_indexes += range(3534, 3539)
trash_indexes += range(3578, 3586)

trash_indexes += range(3638, 3641)
trash_indexes += range(1341, 1349)
trash_indexes += range(2835, 2837)

# TODO Восстановление train и %

start = time.time()
raw_train['A_rate'], raw_train['B_rate'] = A_B_rate_restore(raw_train['A_rate'], raw_train['B_rate'], 1000, 12)

raw_train['A_CH4'] = chemical_data_restore(raw_train['A_CH4'], 500, 20, 100, 10)
raw_train['A_C2H6'] = chemical_data_restore(raw_train['A_C2H6'], 500, 20, 100, 14)
raw_train['A_C3H8'] = chemical_data_restore(raw_train['A_C3H8'], 500, 20, 100, 15)
raw_train['A_iC4H10'] = chemical_data_restore(raw_train['A_iC4H10'], 500, 20, 100, 11)
raw_train['A_nC4H10'] = chemical_data_restore(raw_train['A_nC4H10'], 500, 20, 100, 11)
raw_train['A_iC5H12'] = chemical_data_restore(raw_train['A_iC5H12'], 500, 20, 100, 7)
raw_train['A_nC5H12'] = chemical_data_restore(raw_train['A_nC5H12'], 400, 20, 100, 9)
raw_train['A_C6H14'] = chemical_data_restore(raw_train['A_C6H14'], 500, 20, 100, 18)

end = time.time()
print(end - start)

raw_train = restore_percent(raw_train)

#%%
# Удаляем trash index, делим data на train target
final_train = raw_train.copy()
final_targets = raw_targets.copy()

data = pd.concat([final_train, final_targets], axis=1)
data.drop(list(set(trash_indexes)), axis=0, inplace=True)
data = data.reset_index(drop=True)

final_train = data.iloc[:, :10].copy()
final_targets = data.iloc[:, 10:].copy()

# TODO DROP Индекс timestamp
#  isna()=0, final_train.shape(4112, 10), final_targets.shape(4112, 5)
data = pd.concat([final_train, final_targets], axis=1)
data = data.drop([1353, 1354, 1355, 1437], axis=0)

# final_train = data.iloc[:, :11]
# final_targets = data.iloc[:, 11:]

final_train = data.iloc[:, :11]  # оставили timestamp
final_targets = data.iloc[:, 10:]  # оставили timestamp

# final_train['timestamp'] = final_train['timestamp'].astype('datetime64[ns]')  # str to datetime64
# final_targets['timestamp'] = final_targets['timestamp'].astype('datetime64[ns]')  # str to datetime64

# TODO comb_data train
comb_data = pd.merge(final_train, train, on='timestamp', how='left')

# Индексы Не нулевых A_rate_y
idx = comb_data[~comb_data['A_rate_y'].isnull()].index

comb_train = final_train.iloc[:, :11].copy()

#  Переставим в comb_train  timestamp в начало таблицы
cols = comb_train.drop('timestamp', axis=1).columns.to_list()
cols_true = ['timestamp'] + cols
comb_train = comb_train[cols_true]

comb_train.iloc[idx] = comb_data[~comb_data['A_rate_y'].isnull()].iloc[:, 10:]
final_train = comb_train.copy()

final_train['timestamp'] = final_train['timestamp'].astype('datetime64[ns]')  # str to datetime64
final_targets['timestamp'] = final_targets['timestamp'].astype('datetime64[ns]')  # str to datetime64

# final_targets = final_targets.iloc[:, :2]

#%%
# timestamp - переводим в индекс для дальнейшего связывания индексов
final_targets.set_index('timestamp', inplace=True)
final_train.set_index('timestamp', inplace=True)
# Связывание Индексов
final_train.index = final_targets.index

# Возвращаем timestamp из индекса в таблицу
final_targets.reset_index(inplace=True)
final_train.reset_index(inplace=True)

#%%
# TODO Явное Задание Частоты
#  при этом заполняются пропущенные интервалы которые покажет diff().value_counts()
df_tr = final_train.set_index("timestamp")
df_tr = df_tr.asfreq("30min")
df_tr = df_tr.interpolate(method="time")

df_tg = final_targets.set_index("timestamp")
df_tg = df_tg.asfreq("30min")
df_tg = df_tg.interpolate(method="time")

final_train = df_tr.reset_index()
final_targets = df_tg.reset_index()

#%%
# TODO oversampling ⭐❗ timestamp должен быть переведён в индекс
#  isna()=0, final_train.shape(4112, 10), final_targets.shape(4112, 5)
# final_data = pd.concat([final_train, final_targets], axis=1)
# final_data = pd.merge(final_train, final_targets, on='timestamp', how='left')

# Замена  типа object на datetime64[ns]
# final_data['timestamp'] = final_data['timestamp'].astype('datetime64[ns]')
# final_data.set_index('timestamp', inplace=True)

# valid_data = final_data

#%%
# TODO
# valid_data = valid_data.iloc[:, :14]
# # print(valid_data.index)
#
# valid_data = valid_data[final_data.columns]  # Синхронизация
#
# valid_data = valid_data.interpolate()  # На всякий случай


# TODO Для формата etna отделяем train от target
# valid_data = valid_data.reset_index()
#
# final_data = valid_data.copy()

# final_train = final_data.iloc[:, :10]
# final_train.reset_index(inplace=True)
#
# final_targets = final_data.iloc[:, -4:]
# final_targets.reset_index(inplace=True)

# final_test = test.copy()

#%%
# TODO Проверка timestamp у  final_train и final_targets:
#  Одинаковость len размера, equals - совпадение значений,  equals min / max

# Одинаковость размеров
print(len(final_targets))
print(len(final_train))

# Одинаковость границ
print(final_targets["timestamp"].min())
print(final_targets["timestamp"].max())
print(final_train["timestamp"].min())
print(final_train["timestamp"].max())

# Совпадение Значений
final_targets["timestamp"].equals(
    final_train["timestamp"]
)

# Проверка на одинаковость интервалов
mask = final_targets["timestamp"].diff() != pd.Timedelta("30min")
print('Timedelta', final_targets.loc[mask, ["timestamp"]])

# Проверка на одинаковость интервалов
final_train["timestamp"].diff().value_counts().sort_index()
final_targets["timestamp"].diff().value_counts().sort_index()
#%%
# TODO Приводим к формату etna train от target

# final_train.drop(['A_rate', 'B_rate'], axis=1, inplace=True)

df_regressors = final_train.melt(id_vars="timestamp", var_name="segment", value_name="target")
df_to_forecast = final_targets.melt(id_vars="timestamp", var_name="segment", value_name="target")

#%%
# TODO TSDataset
ts = TSDataset(
    df=df_to_forecast,
    freq="30min",
    df_exog=df_regressors
)

#%%
#%%
# TODO EDA До преобразований
ts.plot(figsize=(5, 3))
plt.show()

#%%
acf_plot(ts, lags=10, figsize=(5, 3))
plt.show()

#%%
acf_plot(ts, lags=10, figsize=(5, 3), partial=True)
plt.show()

#%%
# TODO Удаляем Тренд
transform = LinearTrendTransform(in_column="target", poly_degree=2)
ts.fit_transform([transform])

ts.plot(figsize=(5, 3))
plt.show()

#%%
# TODO horizon train_ts, test_ts

# NaiveModel
horizon = 3

train_ts, test_ts = ts.train_test_split(test_size=horizon)

#%%
# TODO Проверка данных
print(len(train_ts.to_pandas()))
print(len(test_ts.to_pandas()))

# print(train_ts.to_pandas().isna().sum())
# print(test_ts.to_pandas().isna().sum())

print(train_ts.to_pandas().isna().any().any())
print(test_ts.to_pandas().isna().any().any())

print(train_ts.to_pandas().isna().sum().sum())
print(test_ts.to_pandas().isna().sum().sum())

# Пропуски по строкам
# df = train_ts.to_pandas()
# df[df.isna().any(axis=1)]

#%%
# TODO pipeline
model = NaiveModel(lag=1)
transforms = [AddConstTransform(in_column="target", value=1)]
pipeline = Pipeline(model, transforms=transforms, horizon=3)
pipeline.set_params(**{"model.lag": 3, "transforms.0.value": 2})
Pipeline(model=NaiveModel(lag=3, ),
         transforms=[AddConstTransform(in_column='target', value=2, inplace=True, out_column=None, )], horizon=3, )
model = NaiveModel(lag=1)
transforms = [
    AddConstTransform(in_column="target", value=1)
]
pipeline = Pipeline(
    model=model,
    transforms=transforms,
    horizon=3
)
pipeline.set_params(
    **{
        "model.lag": 3,
        "transforms.0.value": 2
    }
)
pipeline.fit(train_ts)
forecast_ts = pipeline.forecast()

#%%
# TODO baisline NaiveModel
# model = NaiveModel(lag=1)
#
# model.fit(train_ts)
#
# future_ts = train_ts.make_future(2)
#
# print(future_ts.to_pandas().isna().sum())
# #%%
# forecast_ts = model.forecast(
#     future_ts,
#     prediction_size=horizon
# )

#%%
# TODO MAE MAPE
mae = MAE()

score_mae = mae(y_true=test_ts, y_pred=forecast_ts)

print(score_mae)

mape = MAPE()

score_mape = mape(y_true=test_ts, y_pred=forecast_ts)

print(score_mape)
#%%
plot_forecast(
    forecast_ts=forecast_ts,
    test_ts=test_ts,
    train_ts=train_ts
)
plt.show()

#%%

# horizon = 2
#
# model = SeasonalMovingAverageModel(window=24)
#
# model.fit(train_ts)
#
# future_ts = train_ts.make_future(horizon)
#
# forecast_ts = model.forecast(
#     future_ts,
#     prediction_size=horizon
# )

#%%

# horizon = 48
#
# train_ts, test_ts = ts.train_test_split(test_size=horizon)
#
# # лаги (важно для линейной модели)
# transforms = [
#     LagTransform(in_column="target", lags=[1, 2])
# ]
#
# model = LinearPerSegmentModel()
#
# model.fit(train_ts)
#
# future_ts = train_ts.make_future(horizon)
#
# forecast_ts = model.forecast(future_ts)

#%%
# model = CatBoostPerSegmentModel()
#
# model.fit(train_ts)
#
# future_ts = train_ts.make_future(48)
#
# forecast_ts = model.forecast(future_ts)

#%%
# from etna.transforms import LagTransform, TimeSeriesImputerTransform
# from etna.models import CatBoostPerSegmentModel

# horizon = 48
#
# train_ts, test_ts = ts.train_test_split(test_size=horizon)
#
# transforms = [
#     TimeSeriesImputerTransform(in_column="target"),
#     LagTransform(in_column="target", lags=[1, 2, 3, 4, 5, 6, 12, 24, 48])
# ]
#
# model = CatBoostPerSegmentModel()
#
# model.fit(train_ts, transforms=transforms)
#
# future_ts = train_ts.make_future(horizon)
#
# forecast_ts = model.forecast(future_ts)

#%%

# from etna.transforms import LagTransform
# from etna.models import LinearPerSegmentModel

# horizon = 48
#
# train_ts, test_ts = ts.train_test_split(test_size=horizon)
#
# transforms = [
#     LagTransform(in_column="target", lags=[1, 2])
# ]
#
# model = LinearPerSegmentModel()
#
# model.fit(train_ts, transforms=transforms)
#
# future_ts = train_ts.make_future(horizon)
#
# forecast_ts = model.forecast(future_ts)

#%%
# TODO cv - кастомная валидация
cv = [[np.arange(0 + i * 456, 912 + i * 456), np.arange(0, 4208)] for i in range(8)]
# cv = [[np.arange(0 + i * 456, 912 + i * 456), np.arange(i * 456 + 912, i * 456 + 912 + 456)] for i in range(16)]

# drop_folds = [0, 7, 3, 8, 13]
# cv = [cv[i] for i in range(len(cv)) if i not in drop_folds]

#%%
# TODO target + train + timestamp приводим к Формату  ETNA через melt
# Таблица  растягивается вверх поэтому следим за shape
# data_copy = data.copy()
# data_etna = data_copy.melt(id_vars='timestamp',  var_name="segment", value_name="target")
# data_etna.shape

#%%
# TODO Прогноз для каждого таргета отдельно
# submission = sample.copy()
# submission.iloc[:, 1:] = 0

final_train.drop('timestamp', axis=1, inplace=True)
final_targets.drop('timestamp', axis=1, inplace=True)
#%%

total_loss = 0

for num, target in enumerate(final_targets.columns):

    print(target, end='  ')

    loss = 0

    res = pd.Series(np.zeros(final_train.shape[0]))

    for num, (train_idx, test_idx) in enumerate(cv):
        x_train = final_train.iloc[train_idx]
        x_test = final_train.iloc[test_idx]
        y_train = final_targets.iloc[train_idx][target]
        y_test = final_targets.iloc[test_idx][target].reset_index(drop=True)
        print(x_test.shape, y_test.shape)

        model = Ridge()

        model.fit(x_train, y_train)

        res += model.predict(x_test) / len(cv)

        # submission[target] += model.predict(final_test) / len(cv)

    if target == 'B_C2H6':
        res = exponential_smoothing(res, 0.65)
        # submission[target] = exponential_smoothing(submission[target], 0.65)

    if target == 'B_C3H8':
        res = exponential_smoothing(res, 0.2)
        # submission[target] = exponential_smoothing(submission[target], 0.2)

    if target == 'B_iC4H10':
        res = exponential_smoothing(res, 1)
        # submission[target] = exponential_smoothing(submission[target], 1)

    if target == 'B_nC4H10':
        res = exponential_smoothing(res, 0.35)
        # submission[target] = exponential_smoothing(submission[target], 0.35)

    loss = mean_abs_per_err(y_test, res)

    total_loss += loss
    print(round(loss, 5))

print()
print('total_loss', round(total_loss, 5) / 4)

# '''
# >>>
# B_C2H6  4.15904
# B_C3H8  1.92389
# B_iC4H10  1.26067
# B_nC4H10  1.01302
#
# total_loss 2.0891525
# '''

#%%
submission.to_csv('combinated_version_v_6.csv', index=False)
