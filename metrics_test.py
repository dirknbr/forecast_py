
import unittest
from metrics import *
import numpy as np

class TestDatasets(unittest.TestCase):
  def test_mape(self):
  	y = np.arange(1, 10)
  	pred = y.copy()
  	self.assertEqual(mape(y, pred), 0)

  def test_r2_perfect(self):
  	y = np.arange(1, 10)
  	pred = y.copy()
  	self.assertEqual(r2(y, pred), 1)

  def test_bias(self):
    y = np.arange(1, 10)
    pred = y + 1
    self.assertEqual(bias(y, pred), 1)

if __name__ == '__main__':
  unittest.main()