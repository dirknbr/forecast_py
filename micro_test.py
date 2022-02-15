
import unittest
from datasets import *
from micro import *
import numpy as np

class TestMicroModel(unittest.TestCase):
  def test_fit_and_predict(self):
    data = load_daily_data()
    fh = np.arange(1, 11)
    mic = Micro(k=max(fh))
    mic.fit(data.y)
    pred = mic.predict(fh)
    self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()