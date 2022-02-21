
import unittest
from greyk import *
from datasets import *
import numpy as np

class TestGreykite(unittest.TestCase):
  def test_pandas_data(self):
    data = load_daily_data(100)
    fh = np.arange(1, 11)
    g = Greykite(len(fh))
    g.fit(data)
    pred = g.predict(fh)
    self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()