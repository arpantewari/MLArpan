import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import DateOffset
df=pd.read_csv("monthly-milk-production-pounds.csv")
df.columns=['Month','Milk in pound']
df.drop(168,axis=0,inplace=True)
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month',inplace=True)
df.plot()
time_serios=df['Milk in pound']
time_serios.rolling(12).mean().plot(label='12 month Rolling mean')
time_serios.rolling(12).std().plot(label='12 month Rolling standard deviation')
time_serios.plot()
decomp=seasonal_decompose(time_serios)
fig=decomp.plot()
fig.set_size_inches(15,8)
result=adfuller(df['Milk in pound'])
def adf_check(time_serios):
    result=adfuller(time_serios)
    print("Augemnted  Dicky-Fuller text")
    labels=['ADF Test State','p-value','# of lags','No of Observation']
    for value,labels in zip(result,labels):
        print(labels+" : "+str(value))
    if result[1]>=0.05:
        print("Strong evidence against null hypothesis")
        print("Reject Null hypothesis")
        print("Data have no root value and stationary")
    else:
        print("Weak evidence against null hypothesis")
        print("Fail to Reject Null hypothesis")
        print("Data have root value and non-stationary")
adf_check(df['Milk in pound'])
df['First Difference']=df['Milk in pound']-df['Milk in pound'].shift(1)
df['First Difference'].plot()
adf_check(df['First Difference'].dropna())
df['Second Difference']=df['First Difference']-df['First Difference'].shift(1)
df['Second Difference'].plot()
adf_check(df['Second Difference'].dropna())
df['Seasonality Check']=df['Milk in pound']-df['Milk in pound'].shift(12)
adf_check(df['Seasonality Check'].dropna())
df['Seasonality Check'].plot()
fig_first=plot_acf(df['First Difference'].dropna())
fig_season_first=plot_acf(df['Seasonality Check'].dropna())
autocorrelation_plot(df['Seasonality Check'].dropna())
q=plot_pacf(df['Seasonality Check'].dropna())
model=sm.tsa.statespace.SARIMAX(df['Milk in pound'],order=(0,1,0),seasonal_order=(1,1,1,12))
result1=model.fit()
print(result1.summary())
result1.resid.plot(kind='kde')
future_date=[df.index[-1]+DateOffset(months=x) for x in range(1,24)]
print(future_date)
future_df=pd.DataFrame(index=future_date,columns=df.columns)
print(df.tail())
future_df=pd.DataFrame(index=future_date,columns=df.columns)
final_df=pd.concat([df,future_df])
final_df['forecast']=result1.predict(start=168,end=194)
final_df[['Milk in pound','forecast']].plot()
df['forecast']=result1.predict(start=150,end=168)
df[['Milk in pound','forecast']].plot(figsize=(12,8))
print(df.tail())
plt.show()

