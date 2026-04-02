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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#%%
# todo # Sparse matrices
# * Разрежаем  ручную по столбцам

np.random.seed(1)
mx = np.random.randint(1, 10, size=12).astype('float').reshape(3, 4)

print('Исходная : \n', mx, '\n')

mx[:,  2 * np.arange(2)]= 0

print('Разреженная mx по столбцам  : \n', mx, '\n')

#%%
# todo # Умножение массивов



A = np.array([[2, 0, 7, 2],
              [3, 1, 6, 4],
              [0, 6, 1, 6]])

B = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])

C = A @ B  # или

C_1 = np.dot(A, B)

print(C, '\n')
print(C_1)

#%%
# todo # Определитель ◾ Детерминант

import numpy as np

A = np.array([1, 2, 3, 3, 2, 1, 1, 3, 2]).reshape(3, 3)

det_A = np.linalg.det(A)

det_A

#%%
# todo `np.r_` - соединяет (`склеивает`) `различные` структуры в `np` массив
# * `-1:1:6j` - Создаёт равномерную последовательность из 6 чисел от -1 до 1 (6j = 6 `равномерно` `распределённых` точек).
# *  `[0]*3` = [0, 0, 0]
# * `5, 6 `— просто добавленные элементы.

import numpy as np

print(np.r_[np.array([1, 2, 3]), 0, 0, np.array([4, 5, 6])])
print(np.r_[-1:1:6j, [0]*3, 5, 6])

#%%
# todo # Матричное умножение

d = np.array([10, 20, 30, 40]).reshape(2, 2)
b = np.array([1, 2, 3, 4]).reshape(2, 2)

print('Поэлементное Умножение : \n', b * d)

print('Матричное Умножение, Строки на столбцы \n', b @ d)

#%%
# todo # numpy - char
# * Применять через char методы для строчных значений

# 🔹 Конкатенация
ar = np.array(['a', 'b', 'c'])
np.char.add(ar, 'X')      # ['aX' 'bX' 'cX']
np.char.add('Y', ar)      # ['Ya' 'Yb' 'Yc']

# 🔹 Повторение
np.char.multiply(ar, 3)   # ['aaa' 'bbb' 'ccc']

# 🔹 Изменение регистра
np.char.upper(ar)         # ['A' 'B' 'C']
np.char.lower(ar)         # ['a' 'b' 'c']
np.char.capitalize(ar)    # ['A' 'B' 'C']
np.char.title(['hello world'])  # ['Hello World']

# 🔹 Обрезка и очистка
arr = np.array(['  hi ', ' test  '])
np.char.strip(arr)        # ['hi' 'test']      (обрезка пробелов)
np.char.lstrip(arr)       # ['hi ' 'test  ']
np.char.rstrip(arr)       # ['  hi' ' test']

# 🔹 Поиск и замена
txt = np.array(['apple', 'banana', 'cherry'])
np.char.find(txt, 'a')        # [0 1 -1] (индекс первой 'a')
np.char.count(txt, 'a')       # [1 3 0]  (кол-во 'a')
np.char.replace(txt, 'a', 'X')# ['Xpple' 'bXnXnX' 'cherry']

# 🔹 Проверка
np.char.startswith(txt, 'a')  # [ True False False]
np.char.endswith(txt, 'y')    # [False False  True]

# 🔹 Разбиение строк
np.char.split(['a,b,c', 'd,e'])
# [list(['a', 'b', 'c']) list(['d', 'e'])]

#%%
# todo # numpy - generator random char

df = pd.DataFrame([np.where(np.random.randint(0, 2, size=5) == 1, 'M', 'W')])

df

#%%
# 🚀 todo # numpy - mask

np.random.seed(1)
ar = np.random.randint(0, 2, size=12).reshape(3, 4)

print('Исходный ndarray : \n', ar, '\n')

print('Маска : \n', [ar > 0], '\n')

print('ndarray по маске : \n', ar[ar > 0], '\n')

#%%
# todo # numpy - diff - монотонность
# * `np.diff` - Приращение, вычисляет `дискретную разность` вдоль заданной оси
#   * Если все разности `положительные` или `отрицательные` → возрастающая/убывающая `зависимость → монотонная`.
# * `Монотонность` дискретной разница показывает что признак `числовой`

def check_monotonic(series, target, df):
    """
    Проверяет: при группировке df по series, значения среднего target монотонно возрастают или убывают.
    Возвращает True, если монотонно, иначе False.
    """
    means = df.groupby(series)[target].mean().sort_index().values
    diffs = np.diff(means)
    # Проверим, что либо все приращения >=0 (не убывают), либо все <=0 (не возрастают)
    return (np.all(diffs >= 0) or np.all(diffs <= 0))

# Пример использования:
df = pd.DataFrame({
    'feature': [1, 2, 3, 1, 2, 3],
    'price': [100, 150, 200, 110, 160, 210]
})

print(check_monotonic('feature', 'price', df))  # True — среднее растёт с 1→2→3

import numpy as np
import pandas as pd

# Пример: зависимость среднего таргета от категорий
means = pd.Series([5_000_000, 7_000_000, 9_000_000])

diffs = np.diff(means)
print("Разности:", diffs)

is_monotonic = np.all(diffs > 0) or np.all(diffs < 0)
print("Монотонно:", is_monotonic)

means = pd.Series([7_000_000, 4_000_000, 9_000_000])

diffs = np.diff(means)
print("Разности:", diffs)

is_monotonic = np.all(diffs > 0) or np.all(diffs < 0)
print("Монотонно:", is_monotonic)

def is_monotonic(series):
    diffs = np.diff(series)
    return np.all(diffs > 0) or np.all(diffs < 0)

# Применим
print(is_monotonic([1, 2, 3, 4]))  # True
print(is_monotonic([4, 2, 5]))     # False

#%%
# todo # numpy - diff ◾ Ускорение, Скорость
# * diff(x, n=2) - n это число вычитаний,
#   * n=2 - разница предыдущих разниц

import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.DataFrame(np.random.randint(1, 15, size=9).reshape(-1, 3), columns=list("ABC"))

print('Исходный df : \n', df, '\n')

print('Первая разница  : \n', np.diff(df), '\n')

print('Вторая разница  : \n', np.diff(df, 2), '\n')

# todo # numpy - sign
# * Возвращает знаки поэлементно знаки значений

import numpy as np
ar = np.array([-20, 3, 54, -14, 87])

print('Исходный array :\n', ar, '\n')

print('Определение через sign знаков в array :\n', np.sign(ar), '\n')

print('Маска по знакам array :\n', np.sign(ar)==-1, '\n')

print('Вывод значений по Маске array :\n', ar[np.sign(ar)==-1], '\n')

print('Вывод заданных знаков через маску  :\n', np.sign(ar)[np.sign(ar)==-1], '\n')


np.sign(ar)==-1
ar[np.sign(ar)==-1]
np.sign(ar)[np.sign(ar)==-1]

#%%
# todo # 🚀 numpy - скользящее окно window + as_strided
# * `lib.stride_tricks` - модуль numpy
#   * `as_strided` - создаёт представление в массиве с заданно формой и шагами
#     * `представление` - смотрит на ту же `область памяти` по сути `новый` массив на основе `исходного`
#       *  `новый` массив - `любой формы` и `любого размера`
#   * `as_strided` - новый способ `обращения` к `уже существующим` данным.
#     * `безпредельный reshape` ❗
#   * `sliding_window_view` - создаёт окно в массиве с заданной формой окна и шагами
#
# 👉 `as_strided` - оружие `без` предохранителя
# * `as_strided(ar, [12, 2]) `
#   * 12 - [количество шагов](https://)
#   * 2 - [ширина окна](https://)

import numpy as np

np.random.seed(0)
ar = np.arange(1, 25).reshape(6, 4)

print('Исходный ar :\n', ar, '\n')

print(np.lib.stride_tricks.as_strided(ar, [12, 2]), '\n')
print(np.lib.stride_tricks.as_strided(ar, [4, 3]), '\n')

#%%
# todo 🔥 `sliding_window_view`

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

x = np.arange(10)
print(x)
# [0 1 2 3 4 5 6 7 8 9]

windows = sliding_window_view(x, window_shape=3)
print(windows, '\n')

ar = np.array(list('asdfghjklqwe')).reshape(3, 4)

print('Исходный array : \n', ar, '\n')

print('Выпрямленный array : \n', ar.ravel(), '\n')

print(sliding_window_view(ar.ravel(), 3))

pd.DataFrame(sliding_window_view(ar.ravel(), 3))

#%%

# todo `Окно` через цикл

import numpy as np

x = np.arange(10)
window = 3
means = [x[i:i+window].mean() for i in range(len(x)-window+1)]

means

len(x) - window+1

#%%

# todo `Окно` Через `свёртку` (`np.convolve`) — быстрый вариант для `средних`

x = np.arange(10)
window = 3
weights = np.ones(window) / window
moving_avg = np.convolve(x, weights, mode='valid')
print(moving_avg)

#%%
# todo # numpy -  convolve
# `convolve` - возвращает дискретную свёртку 2-х линейных последовательностей

import numpy as np
np.convolve([1, 2, 3], [0, 1, 0.5])

#%%

# 🚀 todo numpy - where
# 👉`np.where` — это векторный [if-else](https://) для NumPy.
#   * поэлементный выбор: `if cond then x else y`
# # np.where
# 👆 должны быть заданы `либо оба`, либо `ни один` из x и y
# *` numpy.where(condition, [x, y,])`
#   * x - замена значений отвечающих условию
#   * y - замена значений неотвечающих условию

import numpy as np

np.random.seed(0)
x = np.random.randint(1, 10, size=12).reshape(3, 4)

print('Исходный array : \n', x, '\n')

print('Значения < 5 заменены на 0, \nЗначения > 5 это  значения из массива х : \n ',
      np.where(x > 5, x, 0), '\n')

print('Значения < 5 умножены на 100 \nЗначения > 5 это значения из массива х : \n ' ,
      np.where(x > 5, x, x * 100), '\n')

print('Значения < 5 заменены на "less", \nЗначения > 5 заменены на "more" : \n ',
      np.where(x > 5, 'more', 'less'), '\n')

#%%
# todo # where с 1D
# * c одним столбцом

import numpy as np

ar = np.array(range(1, 13)).reshape(3, 4)

ar2 = np.array(list('asdfgqwertyg')).reshape(3, 4)

print('Исходный array : \n', ar, '\n')

print('Преобразования в 1-м столбце : \n', np.where(ar[:, 0] == 5, ar[:, 0], 'not equal'), '\n')

print('Исходный array : \n', ar2, '\n')

print('Преобразования в 1-м столбце буквы : \n', np.where(ar2[:, 0] == 'g', ar2[:, 0], 'not equal'), '\n')

print('Преобразования в df буквы : \n', np.where(ar2 == 'g', 'GG', '-'), '\n')


ar2 = np.where(ar2 == 'g', 'GG', 'nan')

ar2

#%%

# todo numpy - дубликаты - np.rec.find_duplicate

import numpy as np

ls = list('asdffawertt')

ls1 = ['asdffawertt']


print('ls : \n', ls, '\n')
print(np.rec.find_duplicate(ls), '\n')

print('ls1 : \n', ls1, '\n')
print(np.rec.find_duplicate(ls1), '\n')

#%%
# todo # numpy - добавить столбец
# 🚀 `column_stack` - соединт array разных размеров

import numpy as np

ar1 = np.arange(0, 12).reshape(3, 4)
ar2 = np.random.randint(100, 1000, size=3)

print('Исходные массивы : \n', ar1, '\n')
print(ar2, '\n')

print('Соединяем через column_stack : \n', np.column_stack((ar1, ar2)))

#%%

# todo numpy - allclose
# * `np.allclose` — это функция для сравнения двух массивов поэлементно с допуском.

import numpy as np

# Из numpy doc:
print(np.allclose([1e10,1e-7], [1.00001e10,1e-8]))

print(np.allclose([1e10,1e-8], [1.00001e10,1e-9]))

print(np.allclose([1.0, np.nan], [1.0, np.nan]))

print(np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True), '\n')


# for example gpt:
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.00000001, 3.0])

print(np.allclose(a, b))

a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.1, 3.0])

print(np.allclose(a, b))

#%%
# 🚀 todo numpy -  select Фильтр
# * `condlist` - список условий
# * `choicelist` - список для выбора
# * 👉 `condlist` и `choicelist` должны иметь одинаковую длину

import pandas as pd
import numpy as np



df = pd.DataFrame({"A": ["F", "F", "W","F", "W", "S", "S", "F", "W"],
              "B": [10, 20, 30, 40, 50, 5 , 15, 60, 25],
              "C": [100, 200, 300, 400, 500, 80, 90, 250, 350],
              "D": ["Avrg", "Bin", "Cnt", "Foo", "Cnt", "Avrg", "Bin", "Cnt", "Avrg"]
                   })

condlist = [np.array(df.B) > 40, np.array(df.B )< 20, np.array(df.B ) == 30]
choicelist = ['more_40', 'less_20', 'aqual_30' ]


select = np.select(condlist, choicelist, default='empty')


print('Фильтр значений  > 40 и < 20 : \n', select)

#%%
# todo # numpy - datetime

import numpy as np
import pandas as pd

dt_rng = pd.date_range('1/1/2020', '31/12/2020',  periods=5) #

print('Исходный DatetimeIndex : \n', dt_rng, '\n')

list(map(lambda x: x.asm8, dt_rng))

#%%
# todo swapaxes - np.transpose
# * `swapaxes` - это `частный` случай `np.transpose` и меняет только` две` оси
# * `np.transpose` - меняет местами `любое количество осей`

import numpy as np

np.random.seed(1)
ar = np.random.randint(1, 10, size=12).reshape(3, 4)
ar1 = np.random.randint(1, 10, size=12).reshape(2, 3, 2)

print('Исходныйй первый : \n', ar, '\n')

print('Исходныйй второй : \n',ar1, '\n')

print('swapaxes : \n', ar.swapaxes(0, 1), '\n')

print('np.transpose-меняем местами две оси: \n', np.transpose(ar, (1, 0)), '\n')

print('np.transpose-меняем местами три оси: \n', np.transpose(ar1, (1, 0, 2)), '\n')
ar1.shape

#%%
# todo loss = a.sum()
# самая простая (прямая) функция потерь, которую можно использовать для проверки или экспериментов.

# 👆  форма (1, 2)  →  значения внутри одной вложенной скобки
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

# 👆  форма (2, 1)   →  каждое значение в отдельной вложенной скобке
w = torch.tensor([[0.5], [0.3]], requires_grad=True)

z = x @ w       # линейная комбинация
a = torch.relu(z)  # функция активации

loss = a.sum()
loss.backward()

#%%
# todo # BGD - MBGD - SGD
# 👉 `torch.optim.SGD `= конкретная реализация `обновления` весов, `оптимизатор`
# 🔥 BGD - MBGD - SGD это способ подачи в оптимизатор
# 👉 Как только выбрали `способ подачи `данных `Dataloader` (batch_size=1, >1, =len(dataset)), фактически задаём `тип градиентного спуска` в смысле `алгоритма`:
# * `batch_size = 1` → `SGD` (стохастический)
# * `batch_size > 1` < len(dataset) → `MBGD` (мини-батчи)
# * `batch_size = len(dataset)` → `BGD` (полный)

import torch
from torch.utils.data import DataLoader, TensorDataset

# Данные
X = torch.randn(10, 5)  # 100 образцов по 10 признаков
y = torch.randn(10, 1)

dataset = TensorDataset(X, y)

# -----------------------------
# Batch Gradient Descent (BGD)
# -----------------------------
# batch_size = весь датасет
loader_bgd = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# -----------------------------
# Mini-batch Gradient Descent
# -----------------------------
# batch_size меньше размера датасета
loader_mbgd = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# Stochastic Gradient Descent (SGD)
# -----------------------------
# batch_size = 1
loader_sgd = DataLoader(dataset, batch_size=1, shuffle=True)

print('mbgd :', loader_mbgd.batch_size)

print('SGD :', list(loader_sgd.batch_sampler))

#%%
# todo Вывод DataLoader
# 1️⃣ Через iter + next

batch = next(iter(loader_mbgd))
print(batch)

# 2️⃣ Через цикл for

print('Вывод x_batch, y_batch : \n', [i for i in loader_mbgd][0])

#%%
# todo # TensorDataset
# * это `обёртка над тензорами`, которая `превращает` один или несколько тензоров в `объект Dataset`
#   * `TensorDataset` не возвращает `данные напрямую`
# * `TensorDataset` нужен для того, чтобы `подготовить` данные к `подаче` в `DataLoader`.
# * Он `обеспечивает индексацию` по образцам и `возможность` составлять `батчи автоматически.`
# * 🔥🔥 `TensorDataset(t1, t2, ..., tn)` все тензоры должны иметь `одинаковое` количество `образцов/объектов/строк` (одинаковый `первый размер`).
#   * Остальные размеры (например, `количество признаков`) могут `отличаться`.

t1 = torch.tensor([[10, 20], [30, 40]])  # shape (2,2)
t2 = torch.tensor([[1, 2], [3, 4]])      # shape (2,2)
dataset = TensorDataset(t1, t2)          # работает, т.к. shape[0] = 2 у обоих

dataset

#%%
# todo  # DataLoader --- TensorDataset(X, y)
# Когда создаём DataLoader через TensorDataset(X, y) то:
# * `Y` уже `встроен` в dataset как `второй тензор`.
# * `DataLoader` просто `разбивает` dataset на `батчи`, возвращая кортеж `(x_batch, y_batch)`.
# * 🔥 `второй элемент` кортежа — это `y_batch`, `батч целей`

dataset = TensorDataset(X, y)
loader_mbgd = DataLoader(dataset, batch_size=2, shuffle=True)
print('Вывод x_batch, y_batch : \n', [i for i in loader_mbgd][0])