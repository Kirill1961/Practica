import random
import pandas as pd
import numpy as np
import re
from datetime import date, datetime, timedelta
import math
import time
from sklearn.linear_model import Ridge
from scipy.signal import find_peaks, peak_prominences

import matplotlib

# matplotlib.use("Agg")
matplotlib.use("TkAgg")
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

import seaborn as sns

# plt.style.use('seaborn')
plt.style.use('seaborn-v0_8-bright')
# plt.style.available

from statsmodels.datasets import co2
from statsmodels.tsa.ardl import ardl_select_order
from statsmodels.datasets.danish_data import load
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL

from etna.datasets import generate_const_df
from etna.datasets import TSDataset

df_tr = pd.read_csv('D:\Eduson_data\sibur_train_features.csv')

df_ts = pd.read_csv('D:\Eduson_data\sibur_test_features.csv')

df_tg = pd.read_csv('D:\Eduson_data\sibur_train_targets.csv')

df_sb = pd.read_csv('D:\Eduson_data\sibur_sample_submission.csv')

#%%
# TODO target
df_tg['timestamp'] = pd.to_datetime(df_tg['timestamp'])

df_tg = df_tg.set_index('timestamp')

pd.infer_freq(df_tg.index)

#%%
# TODO # train sample
#   * train используем как **исходный** df

df_tr['timestamp'] = pd.to_datetime(df_tr['timestamp'])

df_tr = df_tr.set_index('timestamp')

df_tr.shape

#%%
# TODO # Интервалы train sample simple
#  * Смотрим `одинаковость` интервалов
#    * в данном случае все `интервалы = 30 мин`
# 1️⃣
print('Similarity intervals is True :', df_tr.index.diff().nunique(), '\n')

# 2️⃣
df_tr.index.diff().value_counts()

#%%
# TODO # fr Частота train sample
print(pd.infer_freq(df_tr.index), '\n')

# trf(transform) str to int
fr = ''.join([x for x in pd.infer_freq(df_tr.index) if x.isdigit()])

fr = int(fr)

fr

#%%
# TODO # 🚀 Удаление NA
#  * **Только если потребует модель или метод**
#  * Делаем в последнюю очередь тк сбивается сетка freq
# 🚫
df_tr = df_tr.drop('A_rate', axis=1).dropna()

#%%
# TODO # asfreq
#  * если модель не понимает частоту в index, то явно её меняем
import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.DataFrame(
    np.arange(1, 16).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=5, freq='D')
)

print(df, '\n')


# 1️⃣
df = df.asfreq('h')

# 2️⃣ Есля явно не задан индекс
# df.index.freq = 'Н'

# 3️⃣
# df.index.freq = None

# 4️⃣
print('Изменёная частота :', df.index.inferred_freq, '\n')

df

#%%
# TODO # infer_freq - diff()
#  * если `infer_freq` не возвращает частоту то используем `diff()`
import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.DataFrame(
    np.arange(1, 16).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=5, freq='D')
)


df.index.diff().value_counts()

#%%
# TODO # Оценка ряда перед использованием в STL
#  *  если unique ≈ len(data) сигнал не квантованный ≈ не дискретный
#  *  std > 0, чем больше std тем больше вариативность, хорошо для LOESS
#  *  diff.std > 0 → локальная динамика есть
#  *  Если значение залипло - повторяется много раз → плохо
def estimator(data, samples, period: int):
    for sample in samples:

        s = data[sample]
        print("unique:", s.nunique())
        print("std:", s.std())  #
        print("diff std:", s.diff().std())
        print("value counts head:")
        print(s.value_counts().head(), '\n')


estimator(df_tr, df_tr.columns, 48)

#%%
# TODO # 1️⃣ Проверить сезонность **просто**
#  * `std` - если `маленькая вариативность` то разлагать нечего
df_tr.describe()

#%%
# TODO # Обзор визуализация признаков

df_tr.plot()
plt.show()

#%%
# TODO # 2️⃣ Проверить сезонность **просто**
#  * Визуализация по группам (очень мощно)
#  * 👉 Видишь волну → суточная сезонность.
#  👉 Для часовых данных
# df_tr.set_index('timestamp', inplace=True)
df_tg['h'] = df_tg.index.hour
df_tg.groupby('h').mean().plot()
plt.show()

#%%
# TODO fr Частота train sample
print(pd.infer_freq(df_tr.index), '\n')

# trf(transform) str to int
fr = ''.join([x for x in pd.infer_freq(df_tr.index) if x.isdigit()])

fr = int(fr)

fr

#%%
# TODO # Сезонность  seasons STL
#  * Выявление сезонности через STL
#  * Определение `частоты` через **infer_freq** для определения окна в rolling(H)
#  * STL Параметры:
#    * period 👉 длина сезонного цикла
#      * то, что нужно для выявления сезонности
#        * дневные данные, недельная сезонность → period=7
#        * минутные данные, суточная сезонность → period=1440
#    * seasonal
#      * 👉 длина LOESS-сглаживания сезонной компоненты
#      * 👉 должно быть нечётное число ≥ 3
"""
`interpolate()` - По умолчанию удаляет NA только в начале df
* `limit_direction ='both'`
  * **'both'**  - удаляет и в начале и в конце df
  * ‘forward’, ‘backward’ - дибо в начале либо в конце df
"""

print('При freq=30min это 2 ОБ в час = число ОБ в час 60мин / 30мин : \n', int(60 / fr))

print('При freq=30min число наблюдений  в день : \n', int(60 / fr) * 24)

print('При freq=30min число наблюдений  в неделю : \n', int(60 / fr) * 24 * 7 )

print('index : \n', df_tr.index, '\n')

# print('index.freq : \n', df_tr.index.freq, '\n')

print('infer_freq :\n', pd.infer_freq(df_tr.index), '\n')

def stl(data, samples, period: int, plot=False):

    result = {}

    for sample in samples:

        # TODO limit_direction="both" - Заполнение NA в начале и в конце df
        feature = data[sample].astype(float).interpolate(limit_direction='both')

        stl = STL(feature, period=period, robust=True)  # period = предполагаемый сезонный цикл

        res = stl.fit()



        if plot:

          res.plot()

        result[sample] = res

    if plot:

      plt.show()

    return result

f = stl(df_tr, df_tr.columns, 48, plot=True)

#%%
# TODO # Оценка season после STL
#  * STL не отыскивает сезон,
#  * задавая `period= ...` мы формируем гипотезу о размере season,
#  * `period=...` это число наблюдений в сезоне
def estimate_season(data, samples, period: int):

    res_estim = stl(data, samples, period, plot=False)

    for sample in  samples:

        res = res_estim[sample]

        # TODO оценка качества сезонности через std и isna, проверяем наличие сезонности
        print(f'seasonal : {sample} = {res.seasonal.std()}')
        print(f'trend : {sample} = {res.trend.std()}')
        print(f'resid : {sample} = {res.resid.std()}')
        print(f'seasonal.isna : {sample} = {res.seasonal.isna().sum()}')
        print(f'trend.isna : {sample} = {res.trend.isna().sum()}', '\n')

        print(f'***********\n ratio std: {sample} = {res.seasonal.std() / res.resid.std()}')
        print(f' variance explained: {sample} = {1 - np.var(res.seasonal) / np.var(res.seasonal + res.resid)}', '\n')


estimate_season(df_tr, df_tr.columns, 48)

#%%
# TODO # STL doc co2
#  * Четыре графика :
#    *` Observed = Trend + Seasonal + Residual`
register_matplotlib_converters()
data = co2.load_pandas().data
data = data.resample('W').mean().ffill()

res = STL(data).fit()
res.plot()
plt.show()


# TODO CCF
"""
`adjusted` - определяет как считается корреляция между рядами:

  * adjusted=True → используется подгонка под выборку (sample-correlation, unbiased).

  * Это уменьшает смещение, особенно для коротких рядов.

  * adjusted=False → просто обычная корреляция без поправки.

`fft` - как считать CCF

  * fft=True → через Fast Fourier Transform (быстро, особенно для длинных рядов)

  * fft=False → через прямое суммирование (медленно, но иногда точнее при маленьких сериях)

`nlags` - Определяет, сколько лагов по оси X строить.

  * Если nlags=None → автоматически берётся len(x) - 1.

  * Если хочешь посмотреть влияние прошлого до 10 шагов → nlags=10.

alpha - для расчёта доверительного интервала CCF.

  * Если alpha=0.05 → построятся ±95% доверительные интервалы для каждого лага.

  * Полезно, чтобы понять, какие  лаги 👉**статистически значимы**, а какие просто шум.
"""

#%%
# TODO # 👉 preprocessing для CCF
#  * Через ffill / bfill
# конкатим Х и У для удаления NA
x = df_tr.iloc[:, 1]
y = df_tg.iloc[:, 1]

x_y = pd.concat([x, y], axis=1)

# Удаляем NA Через ffill / bfill
x_y.ffill(inplace=True)
x_y.bfill(inplace=True)

# Разделяем Х и У
x = x_y.iloc[:, 0]
y = x_y.iloc[:, 1]

#%%
# TODO # CCF только после удления или Заполнения NA

nlags=10
# nlags=None

alpha=0.05
# alpha=None

ccf_values = ccf(x, y, adjusted=True, fft=True, nlags=nlags, alpha=alpha)

ccf_values






