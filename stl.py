"""
https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html

https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.forecasting.trend.STLForecaster.html
"""

from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA

class STLForecaster:
  def __init__(self, sp=7):
  	self.sp = sp

  def fit(self, y):
  	self.model = STLForecast(y, ARIMA, period=self.sp, model_kwargs={"order": (1, 0, 0)})
  	self.res = self.model.fit()
  	self.y = y

  def predict(self, fh):
  	return self.res.forecast(max(fh))
