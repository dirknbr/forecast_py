"""
Generate some daily data with a trend and weekday pattern

"""

import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.datasets import fetch_openml

def load_daily_data(n=100):
  np.random.seed(22)
  trend = np.arange(n)
  weekday = 2 * (np.arange(n) % 7 == 2)
  y = 100 + weekday + trend ** .5 + np.random.normal(0, 1, n)
  df = pd.DataFrame({'y': y})
  df.index = pd.date_range('2020-01-01', periods=n)
  return df

def load_weekly_data(n=100):
  np.random.seed(22)
  trend = np.arange(n)
  week = 2 * np.isin(np.arange(n) % 52, [11, 24, 37, 50])
  y = 100 + week + trend ** .5 + np.random.normal(0, 1, n)
  df = pd.DataFrame({'y': y})
  df.index = pd.date_range('2020-01-01', periods=n, freq='W')
  return df

def load_zero_bound_data(n=100, p=.1):
  # some of the obs are zero
  np.random.seed(22)
  pos = np.random.binomial(1, (1 - p), n)
  y = np.random.normal(5, 1, n) * pos
  print('0s', np.sum(y == 0))
  df = pd.DataFrame({'y': y})
  df.index = pd.date_range('2020-01-01', periods=n)
  return df

def load_goog_data():
  # https://aroussi.com/post/python-yahoo-finance
  goog = yf.Ticker('GOOG')  
  data = goog.history(start='2020-01-01', end='2021-12-31')
  data['y'] = data.Close
  # data.index.freq = 'D'
  return pd.DataFrame(data['y'])

def load_btc_data():
  # https://aroussi.com/post/python-yahoo-finance
  goog = yf.Ticker('BTC-USD')  
  data = goog.history(start='2020-01-01', end='2021-12-31')
  data['y'] = data.Close
  data.index.freq = 'D'
  return pd.DataFrame(data['y'])

def load_co2_data():
  # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html
  # weekly data
  data = fetch_openml(data_id=41187, as_frame=True).frame
  data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
  data = data[['date', 'co2']].set_index('date')
  return data
