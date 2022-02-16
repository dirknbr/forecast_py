
import unittest
from pop import *
import numpy as np

class TestPopModel(unittest.TestCase):
  def test_sp_1_10(self):
    y = np.arange(1, 11) * 1.
    p = PoPForecaster(1)
    p.fit(y)
    pred = p.predict([1, 2])
    rate = 10 / 9 - 1
    self.assertEqual(pred[0], 10 * (1 + rate))

  def test_sp_3_5(self):
  	# 2 * sp < len(y)
    y = np.array([1, 1, 1, 2, 2])
    p = PoPForecaster(3)
    p.fit(y)
    pred = p.predict([1, 2])
    self.assertEqual(len(pred), 2)
    self.assertEqual(pred[0], 2)

  def test_sp_3_6(self):
  	# 2 * sp >= len(y)
    y = np.array([1, 1, 1, 2, 2, 2])
    p = PoPForecaster(3)
    p.fit(y)
    pred = p.predict([1, 2])
    self.assertEqual(len(pred), 2)
    self.assertEqual(pred[0], 4)

  def test_y_has_0(self):
    y = np.arange(10)
    p = PoPForecaster(3)
    p.fit(y)
    pred = p.predict([1, 2])
    self.assertEqual(len(pred), 2)
    self.assertEqual(pred[0], 9)

if __name__ == '__main__':
  unittest.main()