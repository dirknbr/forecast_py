"""
Log forecaster

reduce outlier impact

https://cadmus.eui.eu/bitstream/handle/1814/11150/MWP_2009_06.pdf?sequence=1&isAllowed=y

x = log(y)
pred_y = exp(x + .5 * sigma ** 2)
"""

import numpy as np
import pandas as pd
from sktime.forecasting.arima import ARIMA

class LogModel:
  def __init__(self, order=(1, 0, 1)):
  	self.order = order

  def fit(self, y):
  	assert y.values.min() > 0
  	x = pd.DataFrame({'x': np.log(y.values)[:, 0]}, index=y.index)
  	self.model = ARIMA(self.order)
  	self.model.fit(x)
  	fh = np.arange(-len(y) + 1, 1)
  	pred_x = self.model.predict(fh)
  	res = (x - pred_x)
  	self.sigma2 = np.mean(res ** 2)

  def predict(self, fh):
  	pred_x = self.model.predict(fh)
  	pred_y = np.exp(pred_x + .5 * self.sigma2)
  	return pred_y
