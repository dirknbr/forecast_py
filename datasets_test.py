
import unittest
from datasets import *

class TestDatasets(unittest.TestCase):
  def test_daily_data(self):
    data = load_daily_data(100)
    self.assertEqual(len(data), 100)
    self.assertEqual(len(list(data)), 1)

  def test_weekly_data(self):
    data = load_weekly_data(100)
    self.assertEqual(len(data), 100)
    self.assertEqual(len(list(data)), 1)

  def test_zero_data(self):
    data = load_zero_bound_data(100)
    self.assertEqual(len(data), 100)
    self.assertEqual(len(list(data)), 1)
    self.assertEqual(min(data.y), 0)

  def test_goog_data(self):
    data = load_goog_data()
    # print(data.index)
    self.assertEqual(len(data), 504)
    self.assertEqual(len(list(data)), 1)

  def test_btc_data(self):
    data = load_btc_data()
    self.assertEqual(len(data), 731)
    self.assertEqual(len(list(data)), 1)

  def test_oil_data(self):
    data = load_oil_data()
    self.assertEqual(len(data), 505)
    self.assertEqual(len(list(data)), 1)

  def test_co2_data(self):
    data = load_co2_data()
    self.assertEqual(len(data), 2225)
    self.assertEqual(len(list(data)), 1)

if __name__ == '__main__':
  unittest.main()