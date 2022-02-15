
import numpy as np
import matplotlib.pyplot as plt

def plot_actual_pred(actual, pred):
  """Plot actual and pred, where pred only covers the last 'fh' periods.
    Args:
      actual: actual series or array
      pred: array of predictions
  """
  assert len(actual) >= len(pred)
  pred_series = np.zeros_like(actual) * np.nan
  pred_series[-len(pred):] = pred
  plt.plot(actual, label='actual')
  plt.plot(pred_series, label='forecast')
  plt.legend()
  plt.show()
