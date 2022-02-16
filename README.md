
# Forecasting package

This package has a series of forecasters for univariate time series data. 

It also has a function to combine forecasts into an ensemble.

It mostly uses `sktime` which does a good job of porting R::forecast to Python.

However it has a few extensions

- more datasets
- SGT forecaster
- TFP STS forecaster (really slow)
- Fourier extrapolator
- MicroPrediction fast and slow
- Diffencing and RF multi output
- Period on period growth
- ensembling with worst predictors removed (trimmed)

## Dependencies & references

- https://www.sktime.org/en/stable/get_started.html
- https://github.com/alan-turing-institute/sktime
- https://alkaline-ml.com/pmdarima/
- https://facebook.github.io/prophet/docs/quick_start.html
- https://pydlm.github.io/
- https://github.com/microprediction/timemachines
- https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_001.html
- https://docs.pymc.io/en/v3/pymc-examples/examples/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
- https://github.com/deepcharles/ruptures
- http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression.html
- https://www.unofficialgoogledatascience.com/2017/04/our-quest-for-robust-time-series.html
- https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html
- https://juanitorduz.github.io/gp_ts_pymc3/

## Out of scope

- multivariate or X variables

