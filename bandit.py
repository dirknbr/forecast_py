"""
iterative bandit type learning of 3 forecasters with dynamic weights

forec = [last, ES, AR1]
alpha = [1, 1, 1]

for t in range(n)
  make fh step forecasts
  increment alpha for the best forecast

optional skip outliers
"""

import numpy as np
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

class Bandit:
  def __init__(self, outlier_removal=False, cutoff=.2):
  	self.alpha = np.ones(3)
  	self.outlier_removal = outlier_removal
  	self.cutoff = cutoff

  def fit(self, y):
  	# walk through y and update alpha
  	assert len(y) > 10
  	y = np.array(y)
  	for t in range(10, len(y)):
  	  growth = y[t] / y[t - 1] - 1
  	  if self.outlier_removal and abs(growth) > self.cutoff:
  	  	pass
  	  else:
  	  	# do one step forecast
  	    p1 = y[t - 1]
  	    p2 = ExponentialSmoothing().fit(y[:t]).predict([1])
  	    p3 = ARIMA((1, 0, 0)).fit(y[:t]).predict([1])
  	    e1 = abs(y[t] - p1)
  	    e2 = abs(y[t] - p2)
  	    e3 = abs(y[t] - p3)
  	    winner = np.argmin([e1, e2, e3])
  	    self.alpha[winner] += 1
  	print('alpha', self.alpha)
  	self.y = y

  def predict(self, fh):
  	pred = np.zeros((len(fh), 3))
  	pred[:, 0] = np.repeat(self.y[-1], len(fh))
  	pred[:, 1] = ExponentialSmoothing().fit(self.y).predict(fh).squeeze()
  	pred[:, 2] = ARIMA((1, 0, 0)).fit(self.y).predict(fh).squeeze()
  	w = self.alpha / sum(self.alpha)
  	return pred.dot(w)

