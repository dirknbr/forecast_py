"""
https://github.com/linkedin/greykite

https://linkedin.github.io/greykite/docs/0.3.0/html/pages/model_components/0300_seasonality.html
"""

from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
import pandas as pd

class Greykite:
  def __init__(self, k=2):
    self.k = k

  def fit(self, y):
    assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)
    if isinstance(y, pd.Series): 
      y = pd.DataFrame(y)
    value_col = list(y)[0]
    y['date'] = y.index
    metadata = MetadataParam(time_col='date', value_col=value_col)
    forecaster = Forecaster()
    self.res = forecaster.run_forecast_config(df=y, 
        config=ForecastConfig(model_template=ModelTemplateEnum.SILVERKITE.name,
                              forecast_horizon=self.k,
                              metadata_param=metadata))

  def predict(self, fh):
    assert len(fh) == self.k
    pred = self.res.forecast.df
    return pred.tail(len(fh))['forecast']