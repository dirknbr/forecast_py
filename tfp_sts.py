
import tensorflow_probability as tfp
import numpy as np

class TfpSts:
  def __init__(self, sp=7):
    self.sp = sp

  def fit(self, y, num_results=100, num_warmup_steps=50):
    # https://www.tensorflow.org/probability/api_docs/python/tfp/sts/forecast
    seas = tfp.sts.Seasonal(num_seasons=self.sp, observed_time_series=y, name='seas')
    llt = tfp.sts.LocalLinearTrend(observed_time_series=y, name='trend')
    self.model = tfp.sts.Sum(components=[seas, llt], observed_time_series=y)
    samples, kernel = tfp.sts.fit_with_hmc(self.model, y, 
      num_results=num_results, num_warmup_steps=num_warmup_steps)
    self.y = y
    self.samples = samples

  def predict(self, fh):
    pred = tfp.sts.forecast(self.model, self.y, parameter_samples=self.samples, 
      num_steps_forecast=max(fh).astype(np.int32))
    return pred.mean().numpy()[:, 0]


