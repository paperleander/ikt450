# PLOTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from pandas import Grouper
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
# https://seklima.met.no/observations/

path = 'data/landvik/landvik_2005_2020.csv'
df = pd.read_csv(path, sep=";", decimal=",", encoding="UTF-8",na_values='-')

# Create another column with correct Datetime
df["Date"] = pd.to_datetime(df['Time'], format='%d.%m.%Y')

# Filter dates between 01.01.2010 to 01.01.2019 (10 years)
start_date = pd.to_datetime('01-01-2010')
end_date = pd.to_datetime('31-12-2019')
df = df[df.Date.between(start_date, end_date)]

# Set the new Datetime as the index
df.set_index('Date', inplace=True)


# Only work with Temperatur, drop other columns
df.drop(columns=["Temp(Max)", "Temp(Min)", "Wind(Max)", "Wind(Mid)"], inplace=True)


# Fill in missing values
df.interpolate().reset_index() # Fill missing values

# Get date for saving files
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H:%M:%S")


def info(df):
    # Print debug info
    print("#"*20)
    print(df.info())
    print(df.head(3))
    print("#"*20)

def linePlot(df):
    df.plot()
    #plt.xlabel('Date (2010-2019)')
    plt.ylabel('Temperature')
    plt.show()
    plt.savefig('line'+now+'.png')

def scatterPlot(df):
    df.plot(style='k.')
    #plt.xlabel('Date (2010-2019)')
    plt.ylabel('Temperature')
    plt.show()
    plt.savefig('scatter'+now+'.png')

def histogramPlot(df):
    df.hist()
    plt.xlabel('Temperature')
    plt.ylabel('Density')
    plt.show()
    plt.savefig('histogram'+now+'.png')

def densityPlot(df):
    df.plot(kind='kde')
    plt.xlabel('Temperature')
    plt.ylabel('Density')
    plt.show()
    plt.savefig('density'+now+'.png')

def whiskerPlot(df):
    groups = df.groupby(Grouper(freq='YS'))
    years = pd.DataFrame()
    print(groups.mean())
    for name, group in groups:
        print(group.values[0])
        years[name.year] = [y for x in group.values for y in x]
    years.boxplot()
    plt.show()
    plt.savefig('whisker'+now+'.png')

def yearlyBoxPlot(df):
    df = df.reset_index()
    df["Year"] = df["Date"].apply(lambda x: x.year)

    df.boxplot(column=["Temp"], by="Year")
    plt.xlabel('Years')
    plt.ylabel('Temperature')
    plt.show()
    plt.savefig('yearly'+now+'.png')

def monthlyBoxPlot(df):
    df = df.reset_index()
    df["Year"] = df["Date"].apply(lambda x: x.year)
    df["Month"] = df["Date"].apply(lambda x: x.month)

    df.boxplot(column=["Temp"], by="Month")
    plt.xlabel('Months')
    plt.ylabel('Temperature')
    plt.show()
    plt.savefig('monthly'+now+'.png')

def heatmapPlot(df):
    df.drop(columns=["Time"], inplace=True)

    groups = df.groupby(pd.Grouper(freq='Y'))
    years = pd.DataFrame()
    for name, group in groups:
        years[name.year] = group["Temp"].values[0:364]
    years = years.T
    plt.matshow(years, interpolation=None, aspect='auto')
    plt.xlabel('Days')
    plt.ylabel('Years')
    plt.show()
    plt.savefig('heatmap'+now+'.png')

def lagPlot(df):
    df.drop(columns=["Time"], inplace=True)
    lag_plot(df)
    plt.show()
    plt.savefig('lag'+now+'.png')

def correlationPlot(df):
    df.drop(columns=["Time"], inplace=True)
    autocorrelation_plot(df)
    plt.show()
    plt.savefig('correlation'+now+'.png')


if __name__ == "__main__":
    info(df)
    linePlot(df)
    scatterPlot(df)
    histogramPlot(df)
    densityPlot(df)
    whiskerPlot(df)
    yearlyBoxPlot(df)
    monthlyBoxPlot(df)
    heatmapPlot(df)
    lagPlot(df)
    correlationPlot(df)


