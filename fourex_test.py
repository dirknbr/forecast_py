
import unittest
from fourex import *
from datasets import *

class TestFourex(unittest.TestCase):
  def test_fourex(self):
  	data = load_daily_data(100)
  	fh = np.arange(1, 11)
  	four = FourierExtrap()
  	four.fit(data.y)
  	pred = four.predict(fh)
  	self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()