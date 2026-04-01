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

from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL

#%%
# TODO # reindex
#  * Привести порядок строк к другому DataFrame
df1 = pd.DataFrame({'C': np.arange(10, 40, step=10), 'D': list('qwe')}, index=range(2, -1, -1) )
df = pd.DataFrame({'A': np.arange(1, 4), 'B': list('asd')})

print('Исходный df1 :\n',df1, '\n' )
print('Исходный df :\n',df, '\n' )

print('В df1 индексы привели в соответствие с df : \n', df1.reindex(df.index, axis=0), '\n')
print('в df1 пытаемся в столбцы вставить индексы из df :\n', df1.reindex(df.index, axis=1))

#%%
# TODO # df1.index = df.index
#  * Явная замена индексов в df1 на индексы из df
df1 = pd.DataFrame({'C': np.arange(10, 40, step=10), 'D': list('qwe')}, index=range(2, -1, -1) )
df = pd.DataFrame({'A': np.arange(1, 4), 'B': list('asd')}, index=list('qwe'))

print('Исходный df1 :\n',df1, '\n' )
print('Исходный df :\n',df, '\n' )

df1.index = df.index

df1

#%%
# TODO # isin - фильтр df по категориям
#  * Фильтруем весь df по категориям столбца 'D'
import pandas as pd
import numpy as np


np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.array([2, 2, 10, 10, 10, 55, 55,55, 55, 80]),
        'B': np.random.randint(1, 20, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )


df[df['D'].isin(['S', 'R'])]

#%%
# TODO # Отбор совпадающих или несовпадающих значений в Series - Тильда
import pandas as pd

s = pd.Series(list('asdfg'))
s1 = pd.Series(list('fg'))

print('Исходный Series s :\n', s, '\n')
print('Исходный Series s1 :\n', s1, '\n')

print('Совпадающие значения :\n', s[s.isin(s1)], '\n')
print('Несовпадающие значения :\n', s[~s.isin(s1)], '\n')

#%%
# TODO  # Фильтр с логическими операторами `~` и `&`
import pandas as pd

df = pd.DataFrame({
    'city': ['Moscow', 'Kazan', 'Moscow', 'Perm', 'Kazan', 'Perm'],
    'price': [10, 15, 20, 5, 30, 8],
    'rooms': [1, 2, 3, 1, 2, 1]
})

# 🔸 Условие 1: город не Москва
cond1 = ~df['city'].isin(['Moscow'])

# 🔸 Условие 2: цена больше 10
cond2 = df['price'] > 10

# 🔸 Условие 3: комнат меньше 3
cond3 = df['rooms'] < 3

# 🔸 Комбинируем логические условия
filtered = df[cond1 & cond2 & cond3]

print(filtered)

#%%
# TODO # Тильда по проще
#  *  Выражение с тильдой пишем в  `круглые скобки`
import pandas as pd

df = pd.DataFrame({"A": [5, 2, 2, 4, 5],
                     "B": [50, 20, 50, 40, 50],
                     "C": [300, 300, 300, 400, 300]},
                    index=["Dt", "Bin", "Foo", "Cnt", "Avrg"]
                    )

print('Исходный df : \n', df, '\n')


cd1 = df.B < 50
cd2 = df.A > 2
cd3 = ~df.C  # Тильда


print('df понескольким условиям : \n', df[cd1 & cd2 & cd3], '\n')

print('Фильтруем df c тильдой : \n', df[~(df.A == 2)])  # Перед тильдой Круглые скобки

#%%
# TODO # reset_index
#  * `df.reset_index(drop=True, inplace=True)`
#  * `drop=True` → `не сохраняет` старый `индекс` как `столбец` ❗;
#  * `inplace=True` → делает изменения прямо в df, без создания копии.
df = pd.DataFrame({'A': ~np.arange(1, 5), 'B': list('asdf'), 'C': list(range(1,5))})

print('Исходник : \n', df, '\n')


df = df.drop(index=1, axis=0)

print(
      'Удаление строки по индексу :\n',
      df ,'\n' ,
      '👉Переиндексация через reset_index :\n',
      df.reset_index(drop=True), '\n'
      )

#%%
# TODO # loc + iloc + Series - вывод горизонт 1D - вертикаль 2D
df = pd.DataFrame({'A': ~np.arange(1, 5), 'B': list('asdf'), 'C': list(range(1,5))})

print('Исходник : \n', df, '\n')

print('Вывод вертикаль 1D :\n', df.iloc[1, :],'\n')

print('Вывод горизонталь 2D :\n', df.iloc[1:2, :],'\n')

#%%
# TODO # iloc через имя столбца
df = pd.DataFrame({'A': ~np.arange(1, 5), 'B': list('asdf'), 'C': list(range(1,5))})
print('Исходник : \n', df, '\n')
print('Все строки из столбца B : \n',df.iloc[:]['B'], '\n')
print('Значение из столбца : \n', df.iloc[2]['B'], '\n')

#%%
# TODO # `Маска для 1D` df.columns + isin + any
#  * Вывести имя столбца из df по условию
#   * `df[df=='S'].any()` - `одномерная` маска для `одномерного` `Series` `df.columns`
#  * 🚀 `any()` и `all()` - сворачивают 2D в 1D по заданному `axis`
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.random.binomial(5, 0.5, size=10),
        'B': np.random.binomial(5, 0.7, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )


print('Исходный df :\n', df, '\n')

print('Одномерная маска для одномерного списка имён столбцов :\n', df.columns[df[df == 'S'].any()], '\n')
'''
или
'''
print(df.columns[df.isin(['S']).any()], '\n')
'''
или
'''
print(df.columns[df.apply(lambda x: 'S' in x.values)])
'''
или
'''



print('Наличие w в строках df :\n', df[df.isin(['w']).any(axis=1)], '\n')


print('Наличие w в строках df :\n', df[df.isin(['w']).any(axis=1)], '\n')

s = df.isin(['w']).any()
print('1D маска 👇 столбцы содержащие w :\n ', s[s].index)

#%%
# TODO # pd.mask для 1D и 2D
import pandas as pd
import numpy as np

ar = np.array(range(1, 13)).reshape(3, 4)

df = pd.DataFrame(ar, columns=['A', 'B', 'C', 'D'])

print('Исходный df : \n', df, '\n' )

print('mask для Series : \n', df.B.mask(df.B < 6, 'less'), '\n' )

print('mask для df по условию в Series: \n', df.mask(df.B < 6, 'less'), '\n' )

print('mask для df по условию в df: \n', df.mask(df < 6, 'less'), '\n' )

#%%
# TODO # mask - сложная
#  * Для раздельных df и target
import pandas as pd
import numpy as np


df = pd.DataFrame({'A': [1, pd.NA, 3, 4, 5],
       'B': ['a', 's', 'd', pd.NA, 'g'],
       'target': [0, 1, 1, 0, 1]})

print('Исходный df : \n', df, '\n')

y = df['target']

X = df.drop('target', axis=1)

print(X.shape, y.shape)

# Маска сложная для Х и у
mask = X.notna().all(axis=1) & y.notna()  # &y.isna()

mask1 = X.isna()


print(mask)

print(mask1)

X.A.isna() + X.B.isna()

#%%
# TODO # df.any().any()
#  * `Второй .any()` берёт 1D Series jот 1-го any() и проверяет,
#  * есть ли в ней хотя бы одно `True`
#  * Проверяется Есть ли хотя бы `одно` `True` во всём `DataFrame`?
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.random.binomial(5, 0.5, size=10),
        'B': np.random.binomial(5, 0.7, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )

print('Исходный df :\n', df, '\n')

df.isin(['S']).any().any()

#%%
# TODO # map - Series - format
#  * `format`  - подставляет значения Series в строку
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.random.binomial(5, 0.5, size=10),
        'B': np.random.binomial(5, 0.7, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )

print('Исходный df :\n', df, '\n')


print('Содержание столбца А : \n', df.A.map('numerics {}'.format), '\n')
print('Содержание столбца D : \n', df.D.map('letters {}'.format), '\n')

#%%
# TODO # map - df - any function
#  * `map` -применяет функцию ко всему df
#   * здесь `размер` значений Series через `cast_to_str`
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.random.binomial(5, 0.5, size=10),
        'B': np.random.binomial(5, 0.7, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )

print('Исходный df :\n', df, '\n')

print('Размер cat_feat и num_feat до : \n', df.map(lambda x: len(str(x))))


df[['C', 'D']] = df[['C', 'D']]*2

print('Размер cat_feat  после : \n', (df[['C', 'D']]).map(lambda x: len(str(x))))

df[['A', 'B']] = df[['A', 'B']]*100

print('Размер num_feat после : \n', (df[['A', 'B']]).map(lambda x: len(str(x))))

#%%
# TODO # Столбцы df поменять местами
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.random.binomial(5, 0.5, size=10),
        'B': np.random.binomial(5, 0.7, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )

print('Исходный df :\n', df, '\n')

df = df.loc[:, ['D', 'B', 'C', 'A']]

print('Меняем местами столбцы  А и В явно : \n', df)

print('Меняем местами столбцы  А и В через reindex :\n',
      df.reindex(['D', 'B', 'C', 'A'], axis=1))

#%%
# TODO # cut - Фильтр 🔑 Интервал != Группа
#  `pd.cut`- равные `интервалы` значений
#  Используем cut, когда:
#  * важны абсолютные значения
#  * интервалы имеют смысловую шкалу
#  * работаем с возрастом, доходом, стажем
#  `pd.qcut`	равные `группы` наблюдений
#  * квантили (число наблюдений)
#  `labels` → просто имена интервалов
#    * `['one' < 'two' < 'three']`

import pandas as pd
import numpy as np

np.random.seed(0)
sr = pd.Series(np.random.randint(1, 10, size=9))

print(sr, '\n')

sr1 = pd.cut(sr, 3, labels=[ 'one', 'two', 'three'])

print(sr1, '\n')

print(pd.concat([sr, sr1], axis=1))

#%%
# TODO # 🚀 cut - перевод числового признака в категориальный
#  `interv.cat.categories`
#  * `interv` - Series
#  * `cat` - `CategoricalAccessor`: interv.cat
#  * `categories` - `property`: interv.cat.categories
import pandas as pd
import numpy as np

np.random.seed(0)
sr = pd.Series(np.random.randint(1, 10, size=9))

print(sr, '\n')

itr = pd.cut(sr, 3)

itr_name = pd.cut(sr, 3, labels=['one', 'two', 'three'])

itr_name_retbins = pd.cut(sr, 3, labels=['one', 'two', 'three'], retbins=True )

itr_grp = pd.cut(sr, 3).value_counts().reset_index()

itr_name_grp = pd.cut(
    sr, 3,
    labels=['one', 'two', 'three']
    ).value_counts().reset_index()

sr_bound_category = pd.concat([itr_grp['index'].rename('Bound'),
                                itr_name_grp['index'].rename('Category')],
                               axis=1)


print(itr, '\n')

print('Series с именами категорий : \n', itr_name)


print('Series Интервалов : \n', itr, '\n')


print('Границы интервалов : \n', itr.cat.categories, '\n')

print('Series Интервалов с именами категорий + retbins - средние интервалов : \n',
      itr_name_retbins, '\n')

print('Группировка Интервалов + to_df : \n', itr_grp, '\n')

print('Группировка имён категорий + to_df : \n', itr_name_grp, '\n')


print('🚀 Таблица соответствия категорий интервалам : \n', sr_bound_category)

#%%
# TODO # cut - Интервалы в ручную
#  * Уменьшение freq категориального признака
import pandas as pd
import numpy as np

np.random.seed(0)
sr = pd.Series(np.random.randint(1, 10, size=9))

print('Исходный Series :\n', sr)

# 1️⃣ Интервалы в ручную
print(pd.cut(sr, bins=[0, 3, 5, 8], labels=['first', 'second', 'third']), '\n')

# 2️⃣ Интервалы автоматически bins=3
print(pd.cut(sr, bins=3, labels=['first', 'second', 'third']), '\n')

#%%
# TODO # Столбец - Строка Series
import pandas as pd
import numpy as np

dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

df1 = pd.DataFrame(np.arange(1, 19).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

print('Исходный df : \n', df, '\n')


string = df.iloc[2]

bar = df.iloc[2].to_frame()

print('Строка Series :\n', string, '\n')

print('Строка Series :\n', string.index, string.values, string.name, '\n')

print('Столбец Series :\n', bar, '\n')

print('Столбец Series :\n', bar.index, bar.values, bar.columns, '\n')

print('Строка Series Covert -> df :\n', df.iloc[[2]], '\n')

#%%
# TODO # Столбц из Series через `двойные [ ]` и `transpons`
import pandas as pd
import numpy as np

dt = pd.date_range('2010-01-01', periods=6, freq='m' )

df = pd.DataFrame(np.array(list('wwwsssfffvvvqqqddd')).reshape(-1, 3),
                  columns=list('ABC'), index=dt)

df.iloc[[2]].T

#%%
# TODO # wide - long
#  * melt - развернуть DataFrame из wide в long
import pandas as pd
import numpy as np


wd = pd.DataFrame(
    {
        'id': {0: 1, 1: 2, 2: 3},
        'var':{0: 'A', 1: 'B', 2:'C'},
        'val':{0: 20, 1: 30, 2: 40}

    }
)

lng = wd.reset_index().melt(id_vars="id", var_name="segment", value_name="target")

print('Исходный wide table : \n', wd, '\n')

print('Convert  wide - long через melt : \n', lng, '\n')

print('wide - long через stack : \n', wd.stack().reset_index(name='value'))

#%%
# TODO # long - wide

df_long = pd.DataFrame({'id': [1, 1, 2, 2, 3, 3], 'var': list('ABCABC'), 'val': [1, 2, 3, 4, 5, 6]})



print('Исходный df : \n', df_long, '\n')

# 1️⃣
print(df_long.pivot(index='id', columns='var', values='val'), '\n')

# 2️⃣
print(df_long.set_index(['id','var']).unstack(), '\n')

#%%
# TODO # df append через concat
#  * [[n]] - n в двойных скобках, иначе concat не строк 2D а concat столбцов 1D
""" 
#  df.iloc[[1]] -> 2D
# 
#     A  B  C
#  1  4  5  6
#  
#  df.iloc[1] -> 1D
# 
# A    4
# B    5
# C    6
"""
import pandas as pd
import numpy as np



df = pd.DataFrame(
    np.arange(1, 46).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=15, freq='D')
)

vd = pd.DataFrame()

for n in range(len(df)):
  vd = pd.concat((vd, df.iloc[[n]]), axis=0)

vd

#%%
# TODO # df append чётные / не чётные строки
#  * Заполнение
#   * чётных строк значениями
#   * не чётных None
import pandas as pd
import numpy as np



df = pd.DataFrame(
    np.arange(1, 16).reshape(-1, 3),
    columns=list("ABC")
    )


vd = pd.DataFrame()

none = pd.Series([np.nan]*3, index=df.columns)

j = 0

for n in range((len(df))*2 - 1):
    if n % 2 == 0 and j < len(df):
        print(j)
        vd = pd.concat([vd, df.iloc[[j]]], axis=0)
        j += 1

    else:
        vd = pd.concat([vd, none.to_frame().T], axis=0, ignore_index=True)

print(vd, '\n')



# for j in range(df.shape[0]- 1):
#     vd = pd.concat([vd, df.iloc[[j]]], axis=0)
#     vd = pd.concat([vd, none.to_frame().T], axis=0, ignore_index=True)

# # последняя строка
# valid_data = pd.concat([vd, df.iloc[[-1]]], axis=0)

# vd

#%%
# TODO # Индексы замена на чётные
nidx = np.arange(0, 2 * len(df), 2)

df.index = nidx

df

#%%
# TODO # Series столбец - в df строку
#  * Через `to_frame` и `.Т`
import pandas as pd
import numpy as np



df = pd.DataFrame(
    np.arange(1, 46).reshape(-1, 3),
    columns=list("ABC")
    )

s = pd.Series(np.random.randint(1, 10, size=5))

print(s, '\n')

print(s.to_frame().T)

#%%
# TODO # submission
#  * копия исходного датасета с нулевыми значениями
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [pd.NA, 2], 'b':[3, pd.NA]})

print('Исходный df :\n',df, '\n' )

sub = df.copy()

sub[:] = 0

print('df для submission:\n', sub, '\n' )

#%%
# TODO # merge ◾ how='left'
#  * df и df1 одинаковый freq - day / day
#  * df и df3 разный freq - day / month
import pandas as pd
import numpy as np


df = pd.DataFrame(
    np.arange(1, 16).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=5, freq='D')
)
df.index.name = 'timestamp'

df1 = pd.DataFrame(
    np.arange(10, 160, step=10).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=5, freq='D')
)
df1.index.name = 'timestamp'


df2 = pd.DataFrame(
    np.arange(10, 160, step=10).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=5, freq='ME')
)

df2.index.name = 'timestamp'

print('Мерджим df с одинаковой частотой : \n',
      pd.merge(df, df1, on='timestamp'), '\n')

print('Мерджим df с разной частотой, по индексу левой таблицы : \n',
      pd.merge(df, df2, on='timestamp', how='left'), '\n')

print('Мерджим df с разной частотой, по индексу правой таблицы : \n',
      pd.merge(df, df2, on='timestamp', how='right'), '\n')

#%%
# TODO # merge с одинаковыми именами столбцов
#  * если имена left и right одинаковые то pandas присвоит суффиксы _x и _y
import pandas as pd
import numpy as np


dt = pd.date_range('1/1/2020', periods = 12, freq='ME' )
df = pd.DataFrame({'value': np.random.randint(1, 10, size=12)},
                   index=dt)
df.index.name = 'timestamp'
df1 = pd.DataFrame({'value': np.random.randint(10, 100, size=12)},
                   index=dt)
df1.index.name = 'timestamp'
df.merge(df1, on='timestamp', how='left')

#%%
# TODO # idx_tresh
#  * Индекс строк признаков с NA
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [2, 2, 3, 2, 3],
    "B": [pd.NA, 20, pd.NA, 40, pd.NA],
    "C": [pd.NA, 200, 300, 400, pd.NA]
})


idx_na_A = df.loc[df.A.isna() == True, 'A'].index

idx_na_B = df.loc[df.B.isna() == True, 'B'].index

idx_na_C = df.loc[df.C.isna() == True, 'C'].index

print('Индексы строк с NA в столбце "A" : \n', idx_na_A, '\n')
print('Индексы строк с NA в столбце "B" : \n', idx_na_B, '\n')
print('Индексы строк с NA в столбце "C" : \n', idx_na_C, '\n')

# print(idx_tresh)

#%%
# TODO # idx_tresh - коим  в списке через range
idx_tresh = list()

idx_tresh += range(2, 9)
idx_tresh += range(11, 15)
idx_tresh += range(18, 21)

idx_tresh








