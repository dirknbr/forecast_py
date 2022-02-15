
import unittest
import pandas as pd
import numpy as np
from prophet_model import *
# from datasets import *
# from metrics import *

class TestProphetModel(unittest.TestCase):
  def test_fit_and_predict(self):
    n = 100
    fh = [1, 2]
    data = pd.DataFrame({'x': np.arange(n)})
    data.index = pd.date_range('2020-01-01', periods=n)
    # data = load_daily_data()
    pr = ProphetModel()
    pr.fit(data)
    pred = pr.predict(fh)
    self.assertEqual(len(pred), max(fh))

if __name__ == '__main__':
  unittest.main()