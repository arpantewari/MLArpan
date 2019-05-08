import numpy as np
import pandas as pd
import matplotlib.pyplot as okt
import statsmodels.api as sm
df=sm.datasets.macrodata.load_pandas().data
print(sm.datasets.macrodata.NOTE)
index=sm.tsa.datetools.dates_from_range('1959Q1','2009Q3')
df.index=index
df['realgpd'].plot()
plt.show()
