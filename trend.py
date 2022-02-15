
import numpy as np
from sklearn import linear_model

class TrendForecaster:
  def __init__(self, regressor=linear_model.LinearRegression):
  	self.regressor = regressor()

  def fit(self, y):
  	self.X = np.arange(len(y)).reshape(-1, 1)
  	self.regressor.fit(self.X, y)

  def predict(self, fh):
  	X = (np.array(fh) + max(self.X)).reshape(-1, 1)
  	return self.regressor.predict(X)
