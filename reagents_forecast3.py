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
A_flow → расход газа в точке A
B_flow → расход газа в точке B

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
from etna.metrics import  MAPE
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform

# TODO 1 часть: функции
def mape_loss(y_true, y_pred):
    """
    Функция подсчета MAPE
    """
    return (abs((y_true - y_pred) / y_true)).mean() * 100


def exp_smoothing(series, alpha):
    """
    Функция экспоненциального сглаживания, Рекурсивно.
    Получаем на вход сериес, и последовательно сглаживаем значения с "силой" alpha и сдвигом 1 шаг в перёд
    """
    result = [series[0]]

    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])

    return pd.Series(result)


# TODO Кастомный ffill
def A_B_flow_restore(a_flow, b_flow, window, sigma):
    """
    Кастомный ffill()
    Функция, которая восстанавливает значения A_flow и B_flow
    В окне считаем статистики, затем основываясь на среднеквадратичном отклонении определяем аутлаеры
    Если после этого у нас не хватает только одного значения из пары A_flow и B_flow, то строим регрессию
    И восстанавливаем второе. Если неизвестны оба, заполняем на основе предыдущих значений
    Руками находим значения для окон 20 и 100 через CV
    Размер окон 20 и 100, при infer_freq = 30min, 20 - 10 часов и 100 - 50часов примерно 2 дня,
    * 10 часов локальной динамики
    * 2 суток устойчивого направления
    """
    a_flow = a_flow.copy()
    b_flow = b_flow.copy()

    result_a = [a_flow[0]]
    result_b = [b_flow[0]]

    for n in range(1, len(a_flow)):
        mean_a = np.array(result_a[-20:]).mean()
        resid_a = (pd.Series(result_a[-100:]) - pd.Series(result_a[-100:]).shift(1)).mean()
        mean_a += resid_a
        resid_mean_a = abs(pd.Series(result_a[-window:]) - pd.Series(result_a[-window:]).shift(1)).mean()
        resid_std_a = abs(pd.Series(result_a[-window:]) - pd.Series(result_a[-window:]).shift(1)).std()
        mean_b = np.array(result_b[-20:]).mean()
        resid_b = (pd.Series(result_b[-100:]) - pd.Series(result_b[-100:]).shift(1)).mean()
        mean_b += resid_b

        resid_mean_b = abs(pd.Series(result_b[-window:]) - pd.Series(result_b[-window:]).shift(1)).mean()
        resid_std_b = abs(pd.Series(result_b[-window:]) - pd.Series(result_b[-window:]).shift(1)).std()

        if n < window:
            pass
        else:
            if abs(a_flow[n] - result_a[n - 1]) > resid_mean_a + resid_std_a * sigma:
                a_flow[n] = None

            if abs(b_flow[n] - result_b[n - 1]) > resid_mean_b + resid_std_b * sigma:
                b_flow[n] = None

        if pd.isnull(a_flow[n]) and pd.isnull(b_flow[n]):
            result_a.append(mean_a)
            result_b.append(mean_b)

        elif pd.isnull(a_flow[n]):

            model = Ridge()
            model.fit(np.array(result_b[-3000:]).reshape(-1, 1), result_a[-3000:])
            result_a.append(model.predict(np.array(b_flow[n]).reshape(-1, 1))[0])

            result_b.append(b_flow[n])


        elif pd.isnull(b_flow[n]):

            model = Ridge()
            model.fit(np.array(result_a[-3000:]).reshape(-1, 1), result_b[-3000:])
            result_b.append(model.predict(np.array(a_flow[n]).reshape(-1, 1))[0])

            result_a.append(a_flow[n])


        else:
            result_a.append(a_flow[n])
            result_b.append(b_flow[n])

    return pd.Series(result_a), pd.Series(result_b)


def restore_data(series, window, window_mean, window_resid, sigma):
    """
    Функция, которая восстанавливает значения химических элементов
    В окне считаем статистики, затем основываясь на среднеквадратичном отклонении (волатильности) определяем аутлаеры
    И на основе статистик заполняем пропуски
    """
    series = series.copy()

    result = [series[0]]

    for n in range(1, len(series)):

        mean_level = np.array(result[-window_mean:]).mean()

        resid_level = (pd.Series(result[-window_resid:]) - pd.Series(result[-window_resid:]).shift(1)).mean()

        mean_level += resid_level

        resid_mean = abs(pd.Series(result[-window:]) - pd.Series(result[-window:]).shift(1)).mean()
        resid_std = abs(pd.Series(result[-window:]) - pd.Series(result[-window:]).shift(1)).std()

        if n < window:
            pass
        else:

            if abs(series[n] - result[n - 1]) > resid_mean + resid_std * sigma:
                series[n] = None

        if pd.isnull(series[n]):
            result.append(mean_level)
        else:
            result.append(series[n])

    return pd.Series(result)


# TODO Восстановления процентов
def restore_total_percent(data):
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


# TODO 2 часть: подготовка данных с новыми адаптивными сдвигами

# raw_targets = pd.read_csv('D:/Project_reagents/raw_targets.csv')
# raw_test = pd.read_csv('D:/Project_reagents/raw_test.csv')
# raw_train = pd.read_csv('D:/Project_reagents/raw_train.csv')
#
#
# raw_train.drop('timestamp', axis=1, inplace=True)
# raw_test.drop('timestamp', axis=1, inplace=True)
# # # raw_targets.drop('timestamp', axis=1, inplace=True)
#
#
# data = pd.concat([raw_train, raw_targets], axis=1)
# data = pd.concat([data, raw_test], axis=0).reset_index(drop=True)
#
# data = data[2200:].reset_index(drop=True)
#
# data = pd.concat([data, raw_test], axis=0).reset_index(drop=True)
#
#
# data.to_csv('D:/Project_reagents/data.csv', index=False)

#%%
# TODO dataset
data = pd.read_csv('D:/Project_reagents/data.csv')

data = data.copy()


# TODO features = raw_train + raw_targets + raw_test
features = data.iloc[:, :10]
targets = data.iloc[:, 10:]

# TODO features
start = time.time()

features['A_flow'], features['B_flow'] = A_B_flow_restore(features['A_flow'], features['B_flow'], 500, 10)

features['A_4'] = restore_data(features['A_4'], 500, 20, 100, 9)
features['A_26'] = restore_data(features['A_26'], 400, 20, 100, 10)
features['A_38'] = restore_data(features['A_38'], 500, 20, 100, 14)
features['A_4101'] = restore_data(features['A_4101'], 500, 20, 100, 11)
features['A_4102'] = restore_data(features['A_4102'], 500, 20, 100, 11)
features['A_5121'] = restore_data(features['A_5121'], 400, 20, 100, 8)
features['A_5122'] = restore_data(features['A_5122'], 400, 20, 100, 9)
features['A_614'] = restore_data(features['A_614'], 500, 20, 100, 18)

end = time.time()
print(end - start)

features['B_flow_roll'] = features['B_flow'].rolling(190, min_periods=1).mean()

"""
Дальше происходит процесс адаптивных лагов признаков относительно таргета, так как была найдена
взаимосвязь между напором в трубе и временем через которое газ приходит в точку назначения. 
Обычный сдвиг на константу давал результат хуже, 👉 поэтому сдвиг будет адаптивный 
Поэтому в зависимости от расхода в точке В, мы берем более или менее старые данные из прошлого.
👉 A_flow → расход газа в точке A, B_flow → расход газа в точке B
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

        b_flow = features.iloc[i]['B_flow_roll']  # iloc[i] - Series, ['B_flow_roll'] - значение из Series

        shift = 67 / b_flow * 198

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


data = pd.concat([new_features, new_targets], axis=1).dropna(
    subset=['B_26', 'B_38', 'B_4101', 'B_4102']).reset_index(drop=True)

new_features = data.iloc[:, :11].reset_index(drop=True)
new_targets = data.iloc[:, 11:].reset_index(drop=True)

train = new_features.copy()

# data = pd.concat([raw_train, raw_targets], axis=1)

trash_indexes = list()

trash_indexes += range(0, 190)

trash_indexes += range(1191, 1886)

trash_indexes += range(4523, 4689)

data = data.drop(trash_indexes, axis=0).reset_index(drop=True)

raw_train = data.iloc[:, :10]
raw_targets = data.iloc[:, 10:]

trash_indexes = list()

trash_indexes += raw_train[raw_train['A_38'].isnull()].index.to_list()

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
raw_train['A_flow'], raw_train['B_flow'] = A_B_flow_restore(raw_train['A_flow'], raw_train['B_flow'], 1000, 12)

raw_train['A_4'] = restore_data(raw_train['A_4'], 500, 20, 100, 10)
raw_train['A_26'] = restore_data(raw_train['A_26'], 500, 20, 100, 14)
raw_train['A_38'] = restore_data(raw_train['A_38'], 500, 20, 100, 15)
raw_train['A_4101'] = restore_data(raw_train['A_4101'], 500, 20, 100, 11)
raw_train['A_4102'] = restore_data(raw_train['A_4102'], 500, 20, 100, 11)
raw_train['A_5121'] = restore_data(raw_train['A_5121'], 500, 20, 100, 7)
raw_train['A_5122'] = restore_data(raw_train['A_5122'], 400, 20, 100, 9)
raw_train['A_614'] = restore_data(raw_train['A_614'], 500, 20, 100, 18)

end = time.time()
print(end - start)

raw_train = restore_total_percent(raw_train)


# Удаляем trash index, делим data на train target
final_train = raw_train.copy()
final_targets = raw_targets.copy()

data = pd.concat([final_train, final_targets], axis=1)
data.drop(list(set(trash_indexes)), axis=0, inplace=True)
data = data.reset_index(drop=True)

final_train = data.iloc[:, :10].copy()
final_targets = data.iloc[:, 10:].copy()

# TODO DROP Индекс timestamp
data = pd.concat([final_train, final_targets], axis=1)
data = data.drop([1353, 1354, 1355, 1437], axis=0)

final_train = data.iloc[:, :11]  # оставили timestamp
final_targets = data.iloc[:, 10:]  # оставили timestamp

# TODO comb_data train
comb_data = pd.merge(final_train, train, on='timestamp', how='left')

# Индексы Не нулевых A_flow_y
idx = comb_data[~comb_data['A_flow_y'].isnull()].index

comb_train = final_train.iloc[:, :11].copy()

#  Переставим в comb_train  timestamp в начало таблицы
cols = comb_train.drop('timestamp', axis=1).columns.to_list()
cols_true = ['timestamp'] + cols
comb_train = comb_train[cols_true]

comb_train.iloc[idx] = comb_data[~comb_data['A_flow_y'].isnull()].iloc[:, 10:]
final_train = comb_train.copy()

final_train['timestamp'] = final_train['timestamp'].astype('datetime64[ns]')  # str to datetime64
final_targets['timestamp'] = final_targets['timestamp'].astype('datetime64[ns]')  # str to datetime64

# timestamp - переводим в индекс для дальнейшего связывания индексов
final_targets.set_index('timestamp', inplace=True)
final_train.set_index('timestamp', inplace=True)
# Синхронизация Индексов
final_train.index = final_targets.index

# Возвращаем timestamp из индекса в таблицу
final_targets.reset_index(inplace=True)
final_train.reset_index(inplace=True)


# TODO Явное Задание Частоты
#  при этом заполняются пропущенные интервалы которые покажет diff().value_counts()
df_train = final_train.set_index("timestamp")
df_train = df_train.asfreq("30min")
df_train = df_train.interpolate(method="time")

df_targets = final_targets.set_index("timestamp")
df_targets = df_targets.asfreq("30min")
df_targets = df_targets.interpolate(method="time")

final_train = df_train.reset_index()
final_targets = df_targets.reset_index()


# TODO Edison NaiveModel
df_regressors2 = final_train.melt(id_vars="timestamp", var_name="segment", value_name="target")
df_to_forecast2 = final_targets.melt(id_vars="timestamp", var_name="segment", value_name="target")

ts = TSDataset(
    df=df_to_forecast2,
    freq="30min",
    df_exog=df_regressors2
)

horizon2 = 48

train_ts2, test_ts2 = ts.train_test_split(test_size=horizon2)

model = NaiveModel(lag=1)
model.fit(train_ts2)

future_ts = train_ts2.make_future(future_steps=horizon2,
                                  tail_steps=model.context_size)

forecast_ts = model.forecast(future_ts,
                             prediction_size=horizon2)

mape = MAPE()

score_mape = mape(y_true=test_ts2, y_pred=forecast_ts)

print(score_mape)

loss_mape_total = [0]
for component, loss in score_mape.items():
    if component in ['B_26', 'B_38', 'B_4101', 'B_4102']:
        print(component)
        loss_mape_total += loss
print(loss_mape_total)


#%%
plot_forecast(
    forecast_ts=forecast_ts,
    test_ts=test_ts2,
    train_ts=train_ts2,
    figsize = (7, 3)
)
plt.show()

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
# submission.to_csv('combinated_version_v_6.csv', index=False)
