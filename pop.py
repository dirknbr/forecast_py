"""
Period on period forecaster
"""

import numpy as np

class PoPForecaster:
  def __init__(self, sp=12):
    self.sp = sp

  def fit(self, y):
    # cannot divide by 0 or neg
    if sum(y <= 0) == 0:
      y = np.array(y)
      if self.sp == 1:
        self.growth = y[-1] / y[-2] - 1
      else:
        # average the period-on-period rate
        if len(y) >= 2 * self.sp:
          growth = y[-self.sp:] / y[-2 * self.sp:-self.sp] - 1
        else:
          # if sp = 5 and len = 7: y[-2:] / y[-7:-5] - 1
          start = -(len(y) - self.sp)
          # print(start, y[start:], y[start - self.sp:-self.sp])
          growth = y[start:] / y[start - self.sp:-self.sp] - 1
        self.growth = growth.mean()
      print('growth', self.growth)
    else:
      print('y contains 0 or neg. values, just using last value')
      self.growth = None
    self.y = y


  def predict(self, fh):
    if self.growth is not None:
      pred = np.array(list(self.y) + [0] * len(fh))
      for t in range(len(self.y), len(pred)):
        pred[t] = pred[t - self.sp] * (1 + self.growth)
      return pred[-len(fh):]
    else:
      # last value
      return np.repeat(self.y[-1], len(fh))