
# https://pydlm.github.io/simple_example.html

from pydlm import dlm, trend, seasonality
import warnings

warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

class PyDLMForecaster:
  def __init__(self, seasonality=None):
  	self.seasonality = seasonality

  def fit(self, y):
  	self.model = dlm(y)
  	# https://pydlm.github.io/class_ref.html#trend
  	self.model += trend(name='trend', degree=2)
  	if self.seasonality is not None:
  	  # https://pydlm.github.io/class_ref.html#seasonality
  	  self.model += seasonality(self.seasonality)
  	self.model.fit()

  def predict(self, fh):
  	pred, _ = self.model.predictN(max(fh))
  	return pred

