"""
Differentiate y and then learn a tree/forest based on previous values

RF can predict mulitple outputs in 1 row/sample

"""

from sklearn import ensemble, multioutput
import numpy as np

def lag(x, l=1):
  y = np.roll(x, l)
  y[:l] = np.nan
  return y

def lead(x, l=1):
  y = np.roll(x, -l)
  y[-l:] = np.nan
  return y

def valid(x, y):
  # return the rows that have no nan in x and y
  assert x.shape[0] == y.shape[0]
  idx = []
  for i in range(x.shape[0]):
  	if sum(np.isnan(x[i, :])) == 0 and sum(np.isnan(y[i, :])) == 0:
  	  idx.append(i)
  return idx

class DiffTree:
  def __init__(self, ntree=50, nlags=5, fh=[1, 2]):
  	self.ntree = ntree
  	self.nlags = nlags
  	self.fh = fh

  def fit(self, y):
  	assert len(y) > self.nlags
  	dy = np.diff(y)
  	# we lose one obs in diff
  	X = np.zeros((len(y) - 1, self.nlags))
  	Y = np.zeros((len(y) - 1, len(self.fh)))
  	for i in range(1, self.nlags + 1):
  	  X[:, i - 1] = lag(dy, i)
  	for j in self.fh:
  	  Y[:, j - 1] = lead(dy, j)
  	self.model = multioutput.RegressorChain(ensemble.RandomForestRegressor(self.ntree))
  	idx = valid(X, Y)
  	print('using', len(idx), 'obs')
  	self.model.fit(X[idx, :], Y[idx, :])
  	self.pred = self.model.predict(X[-2:, :])[-1, :]
  	self.prev = y[-1]

  def predict(self, fh):
  	assert len(fh) == len(self.fh)
  	pred = np.zeros(len(fh))
  	# reverse the diff()
  	prev = self.prev
  	for i in fh:
  	  pred[i - 1] = prev + self.pred[i - 1]
  	  prev = pred[i - 1]
  	return pred



