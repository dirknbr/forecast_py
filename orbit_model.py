"""
https://github.com/uber/orbit

https://orbit-ml.readthedocs.io/en/stable/tutorials/quick_start.html
"""

from orbit.models import DLT, LGT

class OrbitModel:
  def __init__(self, sp=1, model='LGT'):
  	if model == 'LGT':
  	  self.model = DLT(seasonality=sp)
  	else:
  	  self.model = LGT(seasonality=sp)

  def fit(self, y):
  	y['ds'] = y.index
  	self.model.fit(df=y)
  	self.y = y

  def predict(self, fh):
  	# create forward dataframe
  	freq = self.y.index.freq
  	# https://stackoverflow.com/questions/45253945/how-to-increment-index-of-type-date
  	df = self.y.tail(len(fh))
  	df = df.shift(len(fh), freq=freq)
  	df['ds'] = df.index 
  	pred = self.model.predict(df)
  	return pred.prediction

