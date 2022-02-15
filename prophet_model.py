
from prophet import Prophet
import pandas as pd

class ProphetModel:
  def __init__(self, daily_seasonality=True):
  	self.model = Prophet(daily_seasonality=daily_seasonality)

  def fit(self, y):
    # print(type(y))
    if isinstance(y, pd.core.series.Series):
      y = pd.DataFrame(y)
    if 'y' not in list(y):
      y.rename(columns={list(y)[0]: 'y'}, inplace=True)
    if 'ds' not in list(y):
      y['ds'] = y.index
    self.model.fit(y)

  def predict(self, fh):
  	future = self.model.make_future_dataframe(periods=max(fh))
  	pred = self.model.predict(future).tail(max(fh))
  	return pred['yhat']
