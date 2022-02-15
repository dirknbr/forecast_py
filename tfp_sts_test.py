
import unittest
from tfp_sts import *
from datasets import *

class TestTfpStsModel(unittest.TestCase):
  def test_fit_and_predict(self):
  	y = load_daily_data(50)
  	fh = np.arange(1, 11)
  	sts = TfpSts()
  	sts.fit(y[:40], 10, 10)
  	pred = sts.predict(fh)
  	self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()