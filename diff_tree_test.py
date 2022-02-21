
import unittest
from diff_tree import *
import numpy as np

class TestMicroModel(unittest.TestCase):
  def test_lag(self):
    x = np.arange(10) * 1.
    y = lag(x, 1)
    self.assertEqual(y[1], 0)

  def test_lead(self):
    x = np.arange(10) * 1.
    y = lead(x, 1)
    self.assertEqual(y[1], 2)

  def test_valid(self):
    x = np.array([[0, np.nan], [0, 0], [0, 0]])
    y = np.array([[0, 0], [0, np.nan], [0, 0]])
    idx = valid(x, y)
    self.assertEqual(idx, [2])

  def test_fit_and_predict(self):
    y = np.arange(100) * 1.
    fh = np.arange(1, 5)
    tree = DiffTree(fh=fh)
    tree.fit(y)
    pred = tree.predict(fh)
    # print(pred)
    self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()