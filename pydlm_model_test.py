
import unittest
import numpy as np
from pydlm_model import *

class TestDatasets(unittest.TestCase):
  def test_pydlm(self):
  	y = np.arange(100)
  	fh= np.arange(1, 11)
  	model = PyDLMForecaster()
  	model.fit(y)
  	pred = model.predict(fh)
  	self.assertTrue(len(pred), len(fh))


if __name__ == '__main__':
  unittest.main()