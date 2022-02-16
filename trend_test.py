
import unittest
from trend import *

class TestTrend(unittest.TestCase):
  def test_trend(self):
    y = 1 + np.arange(100)
    fh = np.arange(1, 11)
    tr = TrendForecaster()
    tr.fit(y)
    pred = tr.predict(fh)
    self.assertEqual(len(pred), len(fh))
    self.assertAlmostEqual(pred[0], 101)

if __name__ == '__main__':
  unittest.main()