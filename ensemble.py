"""
Forecast ensembling

Use a trimmed mean approach where we remove the positive and negative forecast
outliers.
"""

from metrics import *
import pandas as pd

def sel(x, idx):
  try:
  	return x[idx]
  except: # if pandas
  	return x.values[idx]

class Ensemble:
  def __init__(self):
  	self.models = None

  def fit(self, y, train, valid, models, remove=2, metric='mae'):
  	"""
  	Args:
  	  y: the target series.
  	  train: index of train samples
  	  valid: index of validation samples, ideally has same length as fh later
  	  models: list of trainable models
  	"""
  	assert len(models) > remove
  	self.models = models
  	fh = valid - max(train)
  	errors = []
  	for model in self.models:
  	  model.fit(sel(y, train))
  	  pred = model.predict(fh)
  	  errors.append(mae(sel(y, valid), pred))
  	print(self.models, errors)
  	# now remove 2 worst ones
  	selected = np.argsort(errors)[:-remove]
  	self.models = [self.models[s] for s in selected]
  	print(self.models)
  	# retrain the models on more data
  	train_new = list(train) + list(valid)
  	for model in self.models:
  	  model.fit(sel(y, train_new))

  def predict(self, fh):
    pred = np.zeros((len(fh), len(self.models)))
    for i, model in enumerate(self.models):
      pred[:, i] = model.predict(fh)[0]
    return pred.mean(axis=1)




