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
#  *   date - конвертирует из объекта даты - строку НО из строки объект нельзя.
#  *   datetime - конвертирует из строки объект дату
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
#  dtm = datetime(2021, 11, 19, 20, 15, 13, 283).strftime('%d   %m-%Y')
#  print(dtm)
#  # Делаем из строки дату
#  s = '2024-07-31 19:09'
#  print(datetime.strptime(s, '%Y-%m-%d %H:%M'))
#  # конвертируем datetime в date
#  print(datetime.strptime(s, '%Y-%m-%d %H:%M').date())
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
# TODO