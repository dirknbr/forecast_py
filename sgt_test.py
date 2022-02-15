
import unittest
from sgt import *
from datasets import *
import numpy as np

class TestSgtModel(unittest.TestCase):
  def test_fit_and_predict(self):
    n = 100
    fh = np.arange(1, 11)
    data = load_daily_data()
    s = SGT()
    s.fit(data, 100, 100, 1)
    pred = s.predict(fh)
    self.assertEqual(len(pred), max(fh))

if __name__ == '__main__':
  unittest.main()