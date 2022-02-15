
# https://gist.github.com/tartakynov/83f3cd8f44208a1856ce

import numpy as np
import pylab as pl
from numpy import fft
    
def fourierExtrapolation(x, n_harm, n_predict):
    n = x.size
    # n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

class FourierExtrap:
  def __init__(self):
  	self.n_harm = 10

  def fit(self, y):
  	self.n = len(y)
  	# TODO: split fit and predict in `fourierExtrapolation`
  	self.pred = fourierExtrapolation(y, self.n_harm, 100)

  def predict(self, fh):
  	return self.pred[self.n:(self.n + max(fh))]


