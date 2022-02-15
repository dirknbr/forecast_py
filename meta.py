"""
Big meta study, cross all the data with all the models

"""

from datasets import *
from metrics import *
from trend import TrendForecaster
from sgt import SGT
from tfp_sts import TfpSts
from prophet_model import ProphetModel
from fourex import FourierExtrap
from recency import RecencyForecaster

import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.croston import Croston


results = []
datas = [load_daily_data(100), load_weekly_data(100), load_zero_bound_data(100), 
         load_btc_data()]
sps = [7, 52, 1, 7]
fh = np.arange(1, 11)
trains = [np.arange(0, 90), np.arange(0, 90), np.arange(0, 90), np.arange(0, 300)]
tests = [np.arange(90, 100), np.arange(90, 100), np.arange(90, 100), np.arange(300, 310)]
models = [NaiveForecaster, 
          TrendForecaster, 
          Croston, 
          ExponentialSmoothing, 
          AutoETS, 
          AutoARIMA, 
          ProphetModel, 
          SGT, 
          ThetaForecaster, 
          # TfpSts(), 
          FourierExtrap, 
          RecencyForecaster
          ]

for i, data in enumerate(datas):
  train = trains[i]
  test = tests[i]
  sp = sps[i]
  for j, model in enumerate(models):
  	if j == 3: # exp smoothing
  	  model_inst = model('add')
  	elif j == 8: # theta
  	  model_inst = model(deseasonalize=False) 
  	else:
  	  model_inst = model()
  	if hasattr(model_inst, 'sp'):
  	  model_inst.sp = sp
  	model_inst.fit(data.y[train])
  	pred = model_inst.predict(fh)
  	# record mae and r2
  	err = mae(data.y[test], np.array(pred))
  	c = corr(data.y[test], np.array(pred))
  	b = bias(data.y[test], pred)
  	results.append((i, model_inst, err, c, b))
  	print(i, model_inst, err, c, b)

df = pd.DataFrame(results)
df.to_csv('results.csv')