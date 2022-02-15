"""
Microprediction

https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/thinking.py

https://github.com/microprediction/timemachines/discussions/14
"""

from timemachines.skaters.simple import thinking
import numpy as np

class Micro:
  def __init__(self, method='slow_and_fast', k=1):
    self.method = method
    # k is the horizon
    # https://github.com/microprediction/timemachines/blob/01acb081a5c9a0f6bf11b4cf21b605cb770d5d57/timemachines/skatertools/composition/residualcomposition.py#L7
    self.k = k

  def fit(self, y):
    s = {}
    for t in range(len(y)):
      pred, _, s = thinking.thinking_slow_and_fast(y[t], s=s, k=self.k)
    # self.s = s
    # self.y = y
    self.pred = pred

  def predict(self, fh):
    # pred, _, _ = thinking.thinking_slow_and_fast(self.y[-1], s=self.s, k=max(fh))
    assert max(fh) == self.k
    return np.array(self.pred)


