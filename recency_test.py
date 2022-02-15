
import unittest
import numpy as np
from recency import *

class TestRecencyModel(unittest.TestCase):
  def test_fit_and_predict(self):
    # y = np.arange(0, 20)
    y = [10] * 20
    fh = np.arange(1, 11)
    rec = RecencyForecaster(lb=3)
    rec.fit(y)
    pred = rec.predict(fh)
    self.assertEqual(sum(pred), 100)

if __name__ == '__main__':
  unittest.main()