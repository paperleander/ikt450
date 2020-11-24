import sys
import pandas as pd
import numpy as np
import datetime


path = 'data/landvik/landvik_2005_2020.csv'
df = pd.read_csv(path, sep=";", decimal=",", encoding="UTF-8",na_values='-')

df["Date"] = pd.to_datetime(df['Time'], format='%d.%m.%Y')
df.drop(columns=["Time", "Temp(Max)", "Temp(Min)", "Wind(Max)", "Wind(Mid)"], inplace=True)

df.interpolate() # Fill missing values

# Filter dates between 01.01.2010 to 31.12.2019 (10 years)
start_date = pd.to_datetime('01-01-2010')
end_date = pd.to_datetime('31-12-2019')
test_start = pd.to_datetime('01-01-2020')
test_end = pd.to_datetime('31-10-2020')

#df= df[df.Date.between(start_date, end_date)]
df= df[df.Date.between(test_start, test_end)]
df.set_index('Date', inplace=True)

#df.to_csv('landvik_2010-2019.csv')
df.to_csv('landvik_2020.csv')


