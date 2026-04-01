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
from etna.datasets import generate_const_df
from etna.datasets import TSDataset

# TODO # ⭐✨ `transform` — это метод объекта `GroupBy`
#  ✅`нет GroupBy нет transform`
#  * `GroupBy` создаётся через df.groupby(...)
#  * Он сам `перебирает` группы, `формирует серии g` и `подаёт их в твою функцию`.
#  * Твоя функция работает `только с одной группой за раз`,
#  а `transform` собирает всё обратно `в результат с тем же размером`, что и `исходный DataFrame`.

#%%
# TODO # 🚀 rolling - Скользящее окно pandas для df
#  * `rolling` - по сути группировка + `agg функции`
#  * `window` - размер скользящего окна
#  * `min_periods` - минимальное количество наблюдений что бы rolling не вернул `NA`
#  * Если проставим `min_periodsint` то pandas просто вставит `исходные строки`
#  * `Вычитание` и `умножение` - через `apply` - `numpy` - `np.diff` / `np.prod`
#  * 🔥 `rolling().apply(func)` — даёт `полный` контроль
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(1, 25).reshape(-1, 4), columns=list('ABCD'))

print('Исходный : \n', df, '\n')

print('Сумма, 1-я ось, окно = 2 :\n', df.rolling(window=2).sum(), '\n')

print('Сумма, 1-я ось, окно = 2, шаг=2 :\n', df.rolling(window=2, step=2).sum(), '\n')

print('Сумма, 2-я ось, окно = 3 :\n', df.rolling(3, axis=1).sum(), '\n')

print('Сумма, 1-я ось, окно = 3 :\n', df.rolling(3, axis=1).sum(), '\n')

print('Сумма, 1-я ось, окно = 2, min_periods=1 :\n', df.rolling(window=2, min_periods=1).sum(), '\n')

print('Вычитание через np.diff :\n', df.rolling(2).apply(lambda x: np.diff(x)), '\n')

print('Умножение через np.prod :\n', df.rolling(2).apply(lambda x: np.prod(x)))

#%%
# TODO # query
#  * Запрос столбцов df с помощью логического выражения, по условию
import pandas as pd
import numpy as np


np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.random.randint(1, 20, size=10),
        'B': np.random.randint(1, 20, size=10),
        'C': list(range(10)),
        'D': np.random.randint(1, 20, size=10)
    }

    )

print('Исходный : \n', df, '\n')

print('Различие между 2-мя столбцами : \n', df.query('A > D'), '\n')

#%%
# TODO # DataFrameGroupBy.rolling
#  * Скользящая группировка
import pandas as pd
import numpy as np


np.random.seed(1)
df = pd.DataFrame(
    {
        'A': [i for i in list(range(1, 4)) * 2],
        'B': np.random.randint(1, 20, size=6),
        'C': list(range(6)),
        'D': np.random.randint(1, 20, size=6)
    }

    )

# df = df.iloc[:5, :]

print('Исходный df : \n', df, '\n')

print('Игнорируем столбец через on : \n', df.groupby('A').rolling(2, on='C').sum(), '\n')

print('Сдвиг по столбцам через min_periods :\n', df.groupby('A')['C'].rolling(2, min_periods=2).sum(), '\n')

#%%
# TODO # index.rename - изменить имя индекса
import pandas as pd
import numpy as np

df = pd.DataFrame(np.round(np.random.rand(16), 3).reshape(-1, 4),
                  columns=list('ABCD'), index=list('abcd'))


print('Исходник : \n', df, '\n')

df.index = df.index.rename('ind')

print('Изменение через index.rename : \n', df, '\n')

df.index = df.index.rename(' ')

df

#%%
# TODO # datetime  - datetime64[ns]
#  Для `datetime64[ns]` `(Timestamp)`:
#  Доступны календарные компоненты:
#  * `.dt.year`
#  * `.dt.month`
#  * `.dt.day` ← это день месяца
#  * `.dt.hour`
#  * `.dt.weekday`
#  ❗❗ `.dt.day ≠ .dt.days.`
import numpy as np
import pandas as pd


t = pd.Series(pd.to_datetime([10, 2, 5]))

print('Исходный datetime : \n', t, '\n')

print('Преобразования через dt.date  : \n', t.dt.date, '\n')

#%%
# TODO # Series.dt.days - timedelta64 - `'s' на конце это интервал`
#  ☝ `timedelta` — это `не дата`, а именно `промежуток` времени
#  * `dt.days` - это компонента `разницы` во времени,  * `dt.days` работает только для `timedelta64`
#  👉Доступны компоненты длительности:
#  * `.dt.days` ← `количество дней интервала`
#  * `.dt.seconds`
#  * `.dt.microseconds`
#  * `.dt.total_seconds()`❗❗ `.dt.day ≠ .dt.days.`

import numpy as np
import pandas as pd



t = pd.Series(pd.to_timedelta([10, 2, 5], unit ='d'))

print('Исходный datetime : \n', t, '\n')

print('Число дней для каждого элемента dt.days : \'n', t.dt.days)

#%%
# TODO # 🚀 groupby - Развёрнутый
#  * `group_keys` - параметр, будут ли `ключи` добавляться в `index`
#    * `group_keys` - по умолчанию True
import random
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

print('Исходный df : \n', df, '\n')

# print('Аналог value_counts() без индексов "A" :\n',
#       df.groupby('A', group_keys=False).apply(lambda x: x), '\n')

print('Отбор заданного числа строк из группировки по "А" через apply  :\n',
      df.groupby('A', group_keys=True).apply(lambda x: x[:2]), '\n')

df.value_counts()

#%%
# TODO # head - groupby
#  * Извлечение `заданного` числа строк из `группировки`
#  * 👉 `nth` -  берёт строки по индексам внутри группы.
import random
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

print('Через head : \n', df.groupby('A').head(2), '\n')

print('Через count < : \n', df[df.groupby('A').cumcount() < 2], '\n')

print('Через nth: \n', df.groupby('A').nth([1,4]), '\n')

print('Через head + as_index: \n', df.groupby('A', as_index=False).head(2), '\n')

print('👍 Через head + as_index: \n', df.groupby('A').head(2).set_index('A', append=True), '\n')

#%%
# TODO # Групповое Кодирование
#  * через LabelEncoder - apply
import random
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder


np.random.seed(1)
df = pd.DataFrame(
    {
        'A': np.array([2, 2, 10, 10, 10, 55, 55,55, 55, 80]),
        'B': np.random.randint(1, 20, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )

print('Исходный df : \n', df, '\n')


cat = df.select_dtypes(include='object').columns

print('Имена cat_feat : \n', cat, '\n')

enc = OrdinalEncoder(dtype=np.int64)

df[cat] = enc.fit_transform(df[cat])

df

#%%
# TODO # to_numpy ◾ get
#  * Преобразование df и Series в nd.array
import random
import pandas as pd
import numpy as np


np.random.seed(2)
df = pd.DataFrame(
    {
        'A': np.arange(0, 10),
        'B': np.random.randint(1, 10, size=10),
        'C': list('awswbwagwa'),
        'D': list('SSPESSRRRw')
    }

    )

print('Исходный df : \n', df, '\n')

df['n'] = np.where(df.B.to_numpy() == 9, 1, 0)

print(df)

df.get('C')

#%%
# TODO
#  contains - имена через маску
#  * Проверяет содержится ли шаблон или регулярное выражение в строке серии или индекса.
#  * pandas - str - contains
import random
import pandas as pd
import numpy as np

srs = pd.Series([
    'numeric_name__AcceptedCmp4', 'numeric_name__AcceptedCmp5',
    'numeric_name__AcceptedCmp1', 'numeric_name__AcceptedCmp2',
    'object_name__Marital_Status_Absurd',
    'object_name__Marital_Status_Alone',
    'numeric_name__MntMeatProducts', 'numeric_name__MntFishProducts',
    'numeric_name__MntSweetProducts', 'numeric_name__MntGoldProds',

])


# 👉 Через маску выводим все имена содержащих 'Mnt'
srs[srs.str.contains(r'Mnt')]

#%%
# TODO # IndexSlice
#  * Срез time series
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.arange(1, 46).reshape(-1, 3),
    columns=list("ABC"),
    index=pd.date_range('01-01-2010', periods=15, freq='D')
)

print('Исходный df : \n', df, '\n')

idx = pd.IndexSlice
row_end = pd.to_datetime(np.percentile(df.index, 90))

# Срез через IndexSlice
slice_90 = df.loc[idx[:row_end]]

slice_10 = df.loc[idx[row_end:]]

print(slice_90, '\n')

print(slice_10, '\n')

#%%
# TODO # time series slice
import pandas as pd
import numpy as np

# df = pd.DataFrame(
#     np.arange(1, 16).reshape(-1, 3),
#     columns=list("ABC"),
#     index=pd.date_range('01-01-2010', periods=5, freq='D')
# )

np.random.seed(0)
dt = pd.date_range(start='1/1/2020', end='12/31/2020', freq='ME')

df = pd.DataFrame({'timestamp': dt, 'target': np.random.randint(1, 100, size=dt.shape[0])})

df['segment'] = 'main'

ts = TSDataset(df, freq='ME')

# Через to_pandas
print(ts.to_pandas().iloc[:3])

# На прямую
ts['2020-04-30': '2020-06-30']

