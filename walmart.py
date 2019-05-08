import pandas as pd
import numpy as np
import matplotlib as plt
df=pd.read_csv("walmart_stock.csv")
df.info()
df['Date']=pd.to_datetime(df['Date'])
print(df['Date'])
