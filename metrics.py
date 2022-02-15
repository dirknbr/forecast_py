"""
Metrics


"""

from sklearn.metrics import r2_score
import numpy as np

def smape(y, pred):
  # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
  return np.mean(abs(pred - y) / (abs(y) + abs(pred)))

def mape(y, pred):
  assert sum(y > 0) == len(y)
  return np.mean(abs(pred - y) / y)

def mae(y, pred):
  return np.mean(abs(y - pred))

def rmse(y, pred):
  return np.sqrt(np.mean((y - pred) ** 2))

def r2(y, pred):
  return r2_score(y, pred)

def corr(y, pred):
  return np.corrcoef(y, pred)[0, 1]

def bias(y, pred):
  # difference in means
  return pred.mean() - y.mean()