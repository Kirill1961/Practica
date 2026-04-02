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

#%%
# TODO # datetime
# *   date - конвертирует из объекта даты - строку НО из строки объект нельзя.
# *   datetime - конвертирует из строки объект дату

import pprint
from datetime import date, datetime, timedelta
import pytz

def convert24_12(time24):
    return datetime.strptime(time24, '%H:%M').strftime('%I:%M%p')


with open('dest_%.csv', 'r') as file:
    ignor = file.readline()
    flights = {}
    for line in file:
        k, v = line.strip().split(',')
        flights[k] = v
        flights2 = {}
for k, v in flights.items():  # items очищает словарь от кавычек и скобок, извлекает из словаря key и value
    flights2[convert24_12(k)] = v.title()
pprint.pprint(flights2)

pprint.pprint({dst: [k for k, v in flights2.items() if dst == v] for dst in flights2.values()})

#%%
# TODO # date - конвертирует из объекта даты - строку НО из строки объект нельзя.

from datetime import date, datetime, timedelta

dt = date(2021, 11, 19)

# Делаем из этой даты строку
print(dt.strftime("%d / %m / %Y"))

# вытаскиваем любые параметры
print(dt.year)
print(dt.month)
print(dt.day)
print(dt.weekday())

# Получаем сегодняшнюю дату
print(dt.today().year)

#%%
# TODO # date - всё таки можно конвертировать из строчных ОБ в ОБ date

from datetime import date

s = ('1981', '11', '10')
date_obj = date(int(s[0]), int(s[1]), int(s[2]))

print(date_obj)

#%%
# TODO # datetime - конвертирует из строки объект дату

from datetime import date, datetime, timedelta
dtm = datetime(2021, 11, 19, 20, 15, 13, 283).strftime('%d   %m-%Y')
print(dtm)

# Делаем из строки дату
s = '2024-07-31 19:09'
print(datetime.strptime(s, '%Y-%m-%d %H:%M'))

# конвертируем datetime в date
print(datetime.strptime(s, '%Y-%m-%d %H:%M').date())

#%%
# TODO from datetime import date, datetime, timedelta
# dtm = datetime(2021, 11, 19, 20, 15, 13, 283).strftime('%d   %m-%Y')
# print(dtm)
# # Делаем из строки дату
# s = '2024-07-31 19:09'
# print(datetime.strptime(s, '%Y-%m-%d %H:%M'))
# # конвертируем datetime в date
# print(datetime.strptime(s, '%Y-%m-%d %H:%M').date())

from datetime import datetime
import pandas as pd

dpt = datetime(2025, 10, 1)  # python

print('Исходный python timstamp :\n', dpt, '\n')

print('Чистая дата через python date() :\n', dpt.date(), '\n')



s = pd.Series(['2025-5-15', '2024-4-14'])

print(s, '\n')

dpd = pd.to_datetime(s)

print('Исходный pandas timstamp, тип не видно если векторный вызов Series :\n', dpd , '\n')

print('Исходный pandas timstamp, тип видно по раздельности :\n', dpd[0], '\n', dpd[1], '\n')

print('pandas timstamp преобразован в pandas datetime.date :\n', dpd.dt.date[0],'\n', dpd.dt.date[1])

#%%
# TODO # datetime -> .strptime() -> .date()
# *   импортируем класс datetime из модуля datetime.
# *   Используем метод strptime() для парсинга строки s по указанному формату '%Y %m %d'.
# *   После этого получаем объект datetime, а затем методом .date() извлекаем объект date с форматом YYYY-MM-DD.

from datetime import datetime

s = '2024 07 31'
date_obj = datetime.strptime(s, '%Y %m %d').date()

print(date_obj)

#%%
# TODO # .strptime()
#  *   datetime.strptime() - конвертирует строку в объект date в виде:
res_format = datetime.strptime(s, "%Y-%m-%d").date()
# возврат >>> [datetime.date(1981, 11, 10), datetime.date(2002, 8, 26), datetime.date(2023, 6, 26)]

#%%
# TODO # .strftime()
# *   .strftime() для форматирования даты
# *   конвертирует в вид '2023-06-26', после преобразования строки в объект date с помощью datetime.strptime()

formatted_date = res_format.strftime("%Y-%m-%d")  # res_format ОБ после .strptime
 # возврат ['1981-11-10', '2002-08-26', '2023-06-26']

 #%%
# TODO  Формат timestamp
#  * Даты и метки времени, метки времени - кол-во секунд с начала ЭПОХИ(01 01 1970)

from datetime import date, datetime, timedelta
s = '2022-11-15 19:09'
dt = datetime.strptime(s, '%Y-%m-%d %H:%M')
ts = datetime.timestamp(dt)
print(ts, "strptime")

# Первод обратно метки времени в дату
print(f"Обратный перевод из метки времени в дату : {datetime.fromtimestamp(ts)}")

#%%
# TODO # pytz.all_timezones - Временные пояса

import pytz
all_time = pytz.all_timezones

# Конструктор часового пояса
time_Mosk = pytz.timezone('Europe/Moscow')
print(time_Mosk)

# данный вывод -  None тк tzinfo ищет часовой пояс но мы его не обозначили
print(datetime(2022, 11, 19, 23, 15, 16, 125).tzinfo, "- tzinfo ищет часовой пояс но мы его не обозначили")
dt = datetime(2022, 11, 19, 23, 15, 16, 125, time_Mosk)
print(dt.tzinfo, " -аргумент pytz.timezone('Europe/Moscow') добавлен")

#%%
# TODO # now() -  Считаем разность дат
# * `now() `- текущая дата
# * `timedelta` - задать интервал дат

from datetime import date, datetime, timedelta
dt = datetime(2022, 11, 19, 23, 15, 16, 125)
print(datetime.now() - dt, "Разность дат")
print(dt + timedelta(days=30), "Сложение дат")

# Разница в секундах
diff = datetime.now() - dt
print(diff.total_seconds() / 60 / 60, "Разница в секундах")

#%%
# TODO # Модуль time
# * например для замера времени работы кода

import time
start = time.time() # time.time()- текущая дата в секундах

print('Hello')

time.sleep(2)  # приостановка, время останова в аргументе

print('Good')

print(f'Итоговое время: {time.time() - start} sec')

#%%
# TODO # Форматы YYYY-MM-DD, Month-Day-YYYY
# Варианты Форматов
# to_char(created_at, 'Month DD, YYYY')

# Если нужно убрать лишние пробелы в названии месяца, можно использовать функцию trim
# trim(to_char(created_at, 'Month')) || ' ' || to_char(created_at, 'DD, YYYY')

# Day — возвращает текстовое название дня недели (например, Monday).
# to_char(created_at, 'Month Day, YYYY')

# комбинирование: числовой день месяца (DD) с названием дня недели (Day)
# to_char(created_at, 'Month DD ("Day": Day), YYYY')

#%%
# TODO Синтетические TS

ts_sr = pd.Series(np.random.randn(1000),
          index=pd.date_range('2020-01-01', periods=1000, freq='h'))

ts_sum = ts_sr.cumsum()

ts_sum.plot()
plt.show()

#%%
# TODO # ts - через Dataframe

np.random.seed(0)

ts_df = pd.DataFrame(np.random.randn(1000, 3),
                      index=ts_sr.index,
                      columns=['one', 'two', 'three'])


ts_df = ts_df.cumsum()

ts_df.plot()

plt.show()

#%%
# TODO ts - через sin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(100)

y = np.sin(t/10) + np.random.normal(0, 0.1, size=100)

plt.plot(t, y)
plt.show()

#%%
# todo shift - pandas
# * freq='A', 'B', 'C', 'D'
# * freq='infer'

import pandas as pd
import numpy as np

dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

df1 = pd.DataFrame(np.arange(1, 19).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')

# for n in range(len(df)):
#   pd.DataFrame(df.shift(periods=n, fill_value=0))


print('fill_value = 0: \n', pd.DataFrame(df.shift(periods=2, fill_value = 0)), '\n')

print('freq="m" : \n', pd.DataFrame(df.shift(periods=2, freq='m')), '\n')

print('freq="infer" : \n', pd.DataFrame(df.shift(periods=2, freq='infer')), '\n')

#%%
# todo 🚀 time seris через data_range + Series
# * **data_range** - задаём как `Индекс`
# * **Series** - задаём наблюдаемые `объекты` на интервале data_range

import random
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import datetime
import matplotlib.pyplot as plt


rn = pd.date_range('1/1/2015', periods=2, freq='h')

ts = pd.Series(np.random.randn(2), index=rn)

# print('time series : \n', ts)

ts

#%%
# todo # resample
# * Преобразование интервала из исходного - в M, D, H, 5Min
# * Перегруппировывает временной ряд по новой частоте и применяет агрегирование.

rn = pd.date_range('1/1/2015', periods=2, freq='h')
ts = pd.Series(np.random.randn(2), index=rn)

ts.resample("5Min").sum()

#%%
# todo Period
# * 👉 freq='M', 'Y' - is deprecated, instead freq='ME', 'YE'
# * Интервал df_ts через Period - 👆 Осторожно

import random
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import datetime
import matplotlib.pyplot as plt


rn = pd.date_range('1/1/2015', periods=7, freq='ME')

p = pd.Period('1/1/2015', freq='M')

ts = pd.DataFrame(np.random.randn(21).reshape(-1, 3), index=rn, columns=list('ABC'))

print('time series : \n', ts)

window = 2

for n in range(-1, len(ts), window):

  # print(n)

  w = (p.month + n, (p + (2 + n)).month)

  under_s = ts[w[0]: w[1]]

  print(under_s)

#%%
# todo # naive model
# * `y_pred = y.shift(1) -> df.shift(1)`
#   * Сдвигаем строки относительно Индекса
# * модель = shift(значений)
# * freq = просто оформление
# ✅ Правило:
# Naive model = shift(H),  # H - горизонт
# НЕ shift(freq=...)
# wwwsssfffvvvqqqddd

import pandas as pd
import numpy as np

dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

df1 = pd.DataFrame(np.arange(1, 19).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')

# print('fill_value = 0: \n', pd.DataFrame(df.shift(periods=2, fill_value = 0)), '\n')


# print('freq="m" : \n', pd.DataFrame(df.shift(periods=2, freq='m')), '\n')

# print('freq="infer" : \n', pd.DataFrame(df.shift(periods=2, freq='infer')), '\n')

print(df.shift(periods=1), '\n')


'''
naive model - прогноз T = month
'''

print('freq="m" - Визуализация : \n ', df.shift(freq='m'), '\n')


# 1️⃣ naive model
# y_targ = df['C']

# y_pred = y_targ.shift(1)

# y_pred.index = y_pred.index + pd.offsets.MonthEnd(1)


# 2️⃣ naive model
# y_targ = df['C']

# print('target : \n', y_targ, '\n')

# y_pred = df['C'].shift(periods=1).shift(freq='M')

# print('Предсказание shift(1)+ shift(freq='"M"') : \n', y_targ)


# 3️⃣ naive model
y_targ = df['C']

y_targ.index = y_targ.index.shift(freq='M')

y_pred = y_targ

print('Предсказание через смещение индексов freq='"M"' : \n', y_pred)


# Добавим предсказанный target
# df['C_pred_targ'] = df['C'].shift(1)

# df = df.drop('C', axis=1).dropna()

# df['lag_1'] = df['C'].shift(1).shift(freq='M')
# df['lag_2'] = df['C'].shift(2).shift(freq='2M')
# df['lag_3'] = df['C'].shift(3).shift(freq='3M')

df

#%%
# todo freq='2Y' - смещение на 2 года
df.shift(freq='2Y')

#%%
# todo # 🚀 direct - Лаги + Целевые
# * `A` и `B` — это **экзогенные** признаки
#   *  их лаги `можно` делать `отдельно`, если нужно.
# * **Лаги** делаем из **исходной** целевой `y=C`
# * `y=C `— это текущее значение цели.
# * `y=C` - нужна только для **генерации** `лагов` и `целевых`
#   *  перед обучением исходную `y=C` **удаляем**

def foo(flag=False):

  dt = pd.date_range('2020-01-01', periods=6, freq='ME' )

  df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                    columns=list('ABC'), index=dt)


  df = df.rename(columns={'C': 'y=C'})

  if flag == True:

    df['lag_1']= df['A'].shift(periods=1)
    df['lag_2']= df['B'].shift(periods=2)

    # df = df.drop('C', axis=1)

    df['y_1'] = df['y=C'].shift(-1)
    df['y_2'] = df['y=C'].shift(-2)


    #  Переставить столбцы через reindex - для красоты, тк 'y=C' удаляется
    df = df.reindex(['A',	'B', 'lag_1',	'lag_2', 'y=C',	'y_1','y_2'], axis=1)

    print('Признаки и Целевые в одном df, Заготовка :\n\n  ', df, '\n')

    df = df.drop('y=C', axis=1).dropna()

    print('Признаки и Целевые в одном df, Готовые к следующим этапам :\n\n  ', df, '\n')

  return df

foo(flag=True)

#%%
# todo # offsets
#
# * `DateOffset / offsets` — это объекты для `календарного сдвига дат`
#
# * сдвиг с учётом `календарной логики`:
#   * index + pd.offsets.Hour(2)
#   * index + pd.offsets.MonthEnd(1)
#
# 👆 `pd.offsets.MonthEnd(1)` - сдвигаем дату на конец следующего месяца
#   * `MonthEnd(1)` - аргумент 1 это сдвиг на 1 месяц
df_init = foo(flag=False)

print('Добавка / сдвиг на 1 месяц с учётом календарной логики : \n', '\n', df_init.index + pd.offsets.MonthEnd(n=1), '\n')

print('Добавка 2 недели с учётом календарной логики : \n', '\n', df_init.index + pd.offsets.Week(n=2), '\n')

print('Добавка 2 часа с учётом календарной логики : \n', '\n', df_init.index + pd.offsets.Hour(2))

#%%
# todo # Timedelta
# * Временная Продолжительность между двумя датами или временными промежутками
df_init = foo(flag=False)

df_init.index + pd.Timedelta(15, 'D')

#%%
# todo # Генерация lag через shift + rolling
# * Генерируем 3 признака:
#   * `lag_A_2, lag_B_4,  rolling_num`
import pandas as pd
import numpy as np

dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

df1 = pd.DataFrame(np.arange(1, 19).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')

df = df.assign(num = np.arange(6))

y_targ = df['C']

df = df.drop('C', axis=1)

y_targ



df['lag_A_2'] = df['A'].shift(freq='2M')

df['lag_B_4'] = df['B'].shift(freq='4M')

df['rolling_num'] = df['num'].rolling(2).mean()

# df = df.asfreq('M')

df
# y_targ

#%%
# todo # WINDOW для undersampling
# * Размер [подвыборки](https://) `60% - 90%` от **train** выборки

import pandas as pd
import numpy as np
import random


dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

# df1 = pd.DataFrame(np.arange(1, 19).reshape(-1, 3),
#                   columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')

samples = []

window = 2

n_batch = 3

stride = 1

idx_samples = list(range(n_batch))

random.shuffle(idx_samples)

for n in range(0, n_batch, stride):
    sample = df.iloc[n:n + window]
    samples.append(sample)


print(idx_samples, '\n')

print(samples)

#%%
# todo windows gpt

window = 6      # длина train окна
stride = 1      # шаг сдвига

for start in range(0, len(df) - window + 1, stride):
    train = df.iloc[start : start + window]

#%%
# todo widow для CV TS

import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(1, 46).reshape(-1, 3), columns=list("ABC"))


# print('Исходный df : \n', df, '\n')

for n in range(5, int(len(df)*0.9), 5):
  print

#%%
# todo expand window для CV
# **SIBUR** через numpy
# * `sliding window`

import pandas as pd
import numpy as np
import random

np.random.seed(0)
dt = pd.date_range('1/1/2020', periods = 12, freq='ME' )

df = pd.DataFrame({'timestamp': dt, 'value': np.arange(12)})

# Sibur, Индексы для кастомной CV на 4 фолда, окно=3,
first_fold = 3
cv = [[np.arange(0 + i * 3, first_fold + i * 3), np.arange(0, 12)] for i in range(4)]

for i in range(len(cv)):
  print(f'idx_train = {cv[i][0]}, idx_test = {cv[i][1]}')

#%%
# todo # 🚀  Генерация Признаков из целевой
# 👉 Фильтр по дате и часу
# * 🚩 Лаги делаем до фильтрации
#   * Фильтруем под нужные моменты прогноза
# * `time series` через `Series + date_range`
# * Если параметры `start и end` то **period** `не указываем`
# * **MS** - month start frequency / явно `задаём частоту` с `начала месяца`
# ✅ Правило:
# rolling(N) = N наблюдений
# rolling('24H') = 24 часа по времени


np.random.seed(0)
dt = pd.date_range(start='1/1/2020', end='12/31/2020', freq='h' )

ts = pd.DataFrame(np.random.randint(1, 100, size=dt.shape[0] * 3).reshape(-1, 3),
                  columns=['x1', 'x2', 'Y'],
                  index=dt
                  )

ts['lag_1'] = ts['Y'].shift(1)
ts['lag_24']  = ts['Y'].shift(24)
ts['roll_mean1'] = ts['Y'].shift(1).rolling('24h').mean()
ts['roll_mean24'] = ts['Y'].shift(24).rolling('24h').mean()
ts['calendar_hour'] = ts.index.month
ts['calendar_day'] = ts.index.day

# Сила тренда
ts['trend_strength'] = ts['Y'].diff().rolling('2h').mean()

# Наклон, скользящее окно с размером 10
ts['slope'] = np.mean(ts['Y'][-10:].diff())


# Фильтр по дате и часу
ts_filtr =  ts.loc[(ts.index.day == 9) & (ts.index.hour == 7)]




print(' 👉 Частота : \n', '👉', ts_filtr.index.freq, '\n')

ts_filtr

#%%
# todo # CCF
# * alpha - доверительный интервал
#   * alpha=0.05 -> 95%
# * nlags - число лагов

x = ts.x1
y = ts.Y

nlags=5
# nlags=None

alpha=0.05
# alpha=None

ccf_values = ccf(x, y, adjusted=True, fft=True, nlags=nlags, alpha=alpha)

ccf_values

#%%
# todo # Целевая для недельного прогноза
# * Прогноз среднего на следующую неделю при freq='D'
#   * `ts['Y'].rolling(7).mean()` - среднее  предыдущих 7 дней
#   * `shift(-7)` - перенос среднего предыдущих 7 дней на будущие 7 дней


print(ts.head(2), '\n')

y_targ = ts['Y'].rolling(7).mean().shift(-7)

y_targ

#%%
# todo # TimedeltaIndex - даты для таргета
# 👆 Способ перейти к конкретному дню следующего месяца.
# * `TimedeltaIndex(..., unit='D')` - Создаёт *для всего ряда* `набор сдвигов` в **днях**.
# * превращает 9-е число текущего месяца
#   * 👉 в 1-е число следующего месяца
#   * 👉 как дату, к которой относится прогноз
# * 📌 day_of_week — это номер дня недели
# * 📌 days_in_month - количество дней в месяце
print((ts.index + pd.TimedeltaIndex(ts.index.days_in_month-8, unit='D')).year)

print((ts.index + pd.TimedeltaIndex(ts.index.day_of_week, unit='D')).day)

# ts.index = ts.index.astype('str')

ts.index.days_in_month

#%%
# todo print((ts.index + pd.TimedeltaIndex(ts.index.days_in_month-8, unit='D')).year)
# print((ts.index + pd.TimedeltaIndex(ts.index.day_of_week, unit='D')).day)
# # ts.index = ts.index.astype('str')
# ts.index.days_in_month


print('Исходные date, 9-е число :\n ', ts.index[:3], '\n')

ts_s = ts.index[:3] + pd.to_timedelta([1, 1, 10], unit='D')

print('Сдвинули 1 и 2 дату на 1 день, 3-ю дату на 10 дней : \n', ts_s, '\n')

print('Альтернатива сдвига через period : \n', ts.index[:3].to_period('M') + 1, '\n')

#%%
# todo # dt
# `Series datetime` → только через **.dt**

ts.index.name = 'id'

ts_1 = ts.reset_index()

ts_1.id[:3] + pd.to_timedelta(ts_1.id[:3].dt.days_in_month, unit='D')

#%%
# todo # asfreq - явно задаём частоту 30 минут

ts = ts.asfreq('30T')
ts

#%%
# todo # asfreq - растянуть ряд с freq='ME'
# * Просто постчитать часы , минуты и тд в ряду с freq='ME'

import numpy as np
import pandas as pd

dt = pd.date_range("2020-01-01", periods=6, freq="ME")
df = pd.DataFrame(np.arange(1, 19).reshape(-1, 3), columns=list("ABC"), index=dt)


print('Исходный df : \n', df, '\n')

print('Уменьшили freq : \n', df.asfreq('h'), '\n')

print('Число часов в ряду : \n', df.asfreq('h').shape[0])

#%%
# todo # Fourier features - Фурье Признаки
# * `df_hour / 24` - трансформируем в сутки
#   * `значение 12 часов / 24 = 0.5 суток`
# * `2 * np.pi * df_hour / 24` = `2π * (t / period)`
#   * переводиv позицию во времени в угол на окружности в радианах
#     * Окружность = 360° = `2π радиан`

import pandas as pd
import numpy as np
import random

np.random.seed(0)
dt = pd.date_range(start='1/1/2020', end='12/31/2020', freq='h' )

sr = pd.Series(np.random.randint(1, 100, size=8761))

df = pd.DataFrame(sr).set_index(dt).rename(columns={0: 'y'})

print('Исходный df : \n', df, '\n')

df_hour = df.index.hour

print('Массив дат в часах : \n', df_hour, '\n')

fourier = np.cos(2 * np.pi * df_hour / 24)

print('Массив косинусов часов : \n', fourier)

#%%
# todo # pipeline для kaggle
# * hour / 24
#   * 👉 переводит час в `долю суточного цикла
# * 2π * hour/24
#   * 👉 переводит `долю суток в угол`
# Координаты точки на окружности
# * x = cos(angle
# * y = sin(angle)

import pandas as pd
import numpy as np
import random

np.random.seed(0)
dt = pd.date_range(start='1/1/2020', end='12/31/2020', freq='h' )

sr = pd.Series(np.random.randint(1, 100, size=8761))

df = pd.DataFrame(sr).set_index(dt).rename(columns={0: 'y'})

print('Исходный df : \n', df, '\n')

# лаги
for l in [1,24,168]:
    df[f'lag_{l}'] = df['y'].shift(l)

# rolling
df['roll_mean_24'] = df['y'].shift(1).rolling(24).mean()

# календарь 👉 преобразуем в часы для cos / sin
df['hour'] = df.index.hour
df['dow'] = df.index.dayofweek

# Fourier
df['sin_24'] = np.sin(2*np.pi*df.index.hour/24)
df['cos_24'] = np.cos(2*np.pi*df.index.hour/24)

df

#%%
# todo # diff в ts
# Yt′  = Yt - Yt-1 приращение Y
# * `Данные станут примерно вокруг одного уровня
# diff(1)  → убирает тренд
# diff(P)  → убирает сезонность с периодом P
# Чаще всего для:
# * ARIMA
# * проверки стационарности
# * удаления тренда

import pandas as pd

df = pd.DataFrame({"a": [1, 0]}, dtype=np.uint8)
df.diff()

np.random.seed(0)
dt = pd.date_range(start='1/1/2020', end='12/31/2020', freq='h' )

ts = pd.DataFrame(np.random.randint(1, 100, size=dt.shape[0] * 3).reshape(-1, 3),
                  columns=['x1', 'x2', 'Y'],
                  index=dt
                  )

print('Исходный ts : \n', ts.head(), '\n')


#'Удаление тренда
ts['diff_1'] = ts['Y'].diff()

# Удаление сезона
ts['diff_24'] = ts['Y'].diff(24)

print('Изменёный  ts : \n', ts.head(), '\n')

#%%
# todo diff в срезе

import pandas as pd
import numpy as np

dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df1 = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

df = pd.DataFrame(np.arange(1, 19).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')

print(df[df['A'].diff().abs() >2], '\n')

print(df[df['A'].diff() != 0], '\n')

df.iloc[:]['B'].diff()

#%%
# todo # diff - аутлаер - выброс - скачок
# * diff()  → разница между соседними значениями
# * abs()   → модуль скачка

import pandas as pd
import numpy as np

np.random.seed(0)
dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.random.randint(1, 15, size=18).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')

#%%
# todo # Найти аномальные скачки
# * df.diff() - скачок от одного к другому
# * jump.mean() - типичное/среднее отклонение
#   * типичная/средняя амплитуда изменения (средний шаг)
# * jump.std() -  волатильность изменения
# * 3 * jump.std() - случайные колебания, шум
#   *  в интервале трёх сигм
# 👉 Алгоритм поиска outlyers:
# 1 diff()      → считаем скачок
# 2 abs()       → берём модуль
# 3 > threshold → проверяем выброс
# 4 loc[...]    → заменяем на NaN

jumps = df.diff().abs()
threshold = jumps.mean() + 3 * jumps.std()
outliers = jumps > threshold

print(outliers)

df[outliers]

#%%
# todo # Размер Периода
# * period - количество наблюдений
# * Визуально через acf
# * Вычислительно через scipy
#   * Вычисляем пики и разницу между ними


