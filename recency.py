
import numpy as np

class RecencyForecaster:
  def __init__(self, lb=7):
  	self.lb = lb

  def fit(self, y):
  	w = np.arange(1, self.lb + 1)
  	self.w = w / sum(w)
  	self.y = y

  def predict(self, fh):
  	pred = np.array(list(self.y) + [0] * len(fh))
  	for t in range(len(self.y), len(pred)):
  	  pred[t] = sum(pred[(t - self.lb - 1):(t - 1)] * self.w)
  	return pred[-len(fh):]
