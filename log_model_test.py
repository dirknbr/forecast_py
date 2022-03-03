
import unittest
from log_model import *
import numpy as np
from datasets import *

class TestLogModel(unittest.TestCase):
  def test_daily(self):
    data = load_daily_data(100)
    # add an outlier
    data.y[10] += 10
    fh = np.arange(1, 11)
    model = LogModel(order=(2, 0, 1))
    model.fit(data)
    pred = model.predict(fh)
    self.assertEqual(len(pred), len(pred))

if __name__ == '__main__':
  unittest.main()