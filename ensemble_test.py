
import unittest
from ensemble import *
from datasets import *
from sktime.forecasting.naive import NaiveForecaster
from trend import TrendForecaster
# from pydlm_model import *
from sktime.forecasting.ets import AutoETS

class TestEnsemble(unittest.TestCase):
  def test_fit_and_predict(self):
    y = load_weekly_data()
    models = [NaiveForecaster(strategy='last'),
              NaiveForecaster(strategy='mean'),
              NaiveForecaster(strategy='drift'),
              TrendForecaster(),
              AutoETS()]
    ens = Ensemble()
    train = np.arange(0, 80)
    valid = np.arange(80, 90)
    fh = np.arange(1, 11)
    ens.fit(y, train, valid, models)
    pred = ens.predict(fh)
    self.assertEqual(len(pred), len(fh))

if __name__ == '__main__':
  unittest.main()