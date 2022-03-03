
from orbit_model import *
from datasets import *
import unittest

class TestOrbitModel(unittest.TestCase):
  def test_daily_dlt(self):
    data = load_daily_data(100)
    fh = np.arange(1, 11)
    orb = OrbitModel(model='DLT')
    orb.fit(data)
    pred = orb.predict(fh)
    self.assertEqual(len(pred), len(pred))

  def test_daily_lgt(self):
    data = load_daily_data(100)
    fh = np.arange(1, 11)
    orb = OrbitModel(model='LGT')
    orb.fit(data)
    pred = orb.predict(fh)
    self.assertEqual(len(pred), len(pred))

if __name__ == '__main__':
  unittest.main()