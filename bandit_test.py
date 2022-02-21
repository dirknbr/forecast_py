
from datasets import *
from bandit import *
import unittest

class TestBanditModel(unittest.TestCase):
  def test_daily_keep_outlier(self):
    data = load_daily_data(50)
    fh = np.arange(1, 5)
    b = Bandit()
    b.fit(data)
    pred = b.predict(fh)
    self.assertEqual(len(pred), len(fh))

  def test_daily_remove_outlier(self):
    data = load_daily_data(50)
    data.y[10] += 10
    fh = np.arange(1, 5)
    b = Bandit(outlier_removal=True)
    b.fit(data)
    pred = b.predict(fh)
    self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()
