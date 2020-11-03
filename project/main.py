# TIME SERIES BASED FORECASTING OF WEATHER IN GRIMSTAD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'data/year.xlsx'
df = pd.read_excel(path, sep=';')

print(df.dtypes)
print(df)

# Create train and test
n_rows = df.shape[0]
split = round(n_rows*0.8)

train = df[0:split]
test = df[split:]


#Plotting data
train.plot(figsize=(15,8), title= 'Daily weather', fontsize=14)
test.plot(figsize=(15,8), title= 'Daily weather', fontsize=14)
plt.show()




# Observed vs Forecasted

