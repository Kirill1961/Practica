import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)       # Увеличивает ширину вывода
from catboost.datasets import amazon
amazon_train, amazon_test = amazon()

print(amazon_train.describe().T)
print(amazon_train.info())

print(amazon_train.nunique(axis=0))

# sns.histplot(amazon_train['ROLE_FAMILY'], kde=True)
# plt.show()