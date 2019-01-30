# Time Series Data

## Metrics:
R-Squared: The percentage of explained variance by the mdoel, this can go from negative infinity
to 1. If it is negative it means that your model was so bad that average outperformed it.

Mean Absolute Error: Interpretable metric, difference between time series prediction and actual time series.

Median Absolute Error: Same as MAE, but with median.

MSE: Differentiable Mean Absolute Error, high penalty to large errors but low penalty to small errors. Sensitive
to outliers.

Mean Squared Log Error: Same as MSE, but take the log of the series, we give more weight to small mistakes as well
(usually used for data with exponential trends).

Mean Absolute Percentage Error: Basically MAE but calculated as a percentage.

Confidence Intervals: If we want to track anomalies in our time series, there are many methods
that we can try to compute the confidence intervals. We basically have an upper confidence and lower
confidence interval. Compute the mean and standard deviation of the time series.

## Smoothing

Instead of weighting the last $k$ values in the time series when computing the running mean, we
can apply an exponentially decaying weighting to previous values.

Triple Exponential Smoothing (Holt Winters): We write out a system of equations which
capture the trend, direction of movement and seasonal components. We decompose the time series.

How do we find the seaonality in our data? Use different window sizes and see which one gives
us a straight line when our running mean is constant.

## Cross Validation in Time Series

TimeSeriesSplit in sklearn - Instead of picking random components of our data, we have an expanding
window where the right hand portion is the test data and the left hand portion is the train data. We
keep expanding and check CV scores.

## Effect of Confidence Intervals

This model can expand its confidence intervals when it sees anomalies so that you don't keep flagging
alarms if there is a constant failure.

# Econometrics Approach

Stationarity: All statistical properties of the time series are preserved no matter which time
window we are looking at (for instance, moments, mean, standard deviation, the correlation
of the time series with itself).

For instance, we may have constant standard deviation in one time series (stationary), but non-constant
standard deviation in another one (not stationary).

We want to ensure that our time series will behave exactly the same as it was in our model, but
if our state is not stationary then that will be a problem for that.

# ARIMA Crash Course

SARIMA(p, d, q)(P, D, Q, s): Seasonal Autoregression Moving Average Model

 - AR(p) - Autoregression model, the regression of the time series onto itself. The basic
   assumption is that the current series values depend on its previous values with some lag. The
   maximum lag is $p$.
 - MA(q) - Moving average part: This models the error out of hte time series

AR(p) + MA(q) = ARMA(p, q): Eliminate error and regress for lag

 - I(d) Order of integration: Number of nonseasonal differences required to make the series stationary.
 - S(s) - Equals season period length of the series.

 We then have three parameter: P, D, Q: P (order of autoregression), Q (similar logic using ACF plots), D
 (order of seasonal integration).

 So for instance, in this example $p$ is most probably 4 since it is the last significant lag on the PACF.
 $d$ equals 1 because we had the first differences. $q$ is somewhere around 4 as see non the ACF.
 $P$ might be 2 since 24th and 48th lags are somewhat significant on the PACF. D equals 1 because we
 performed seasonal differentiation and Q is probably 1.

 We then use the parameter ranges to do a grid search

# Machine Learning for Time Series

All the previous models are okay, but require a lot of tuning.

## Feature Extraction

The model needs features, all we have is a 1-dimensional time series. What can we extract?
 - Lags o the time series
 - Window statistics
   - Min/max value of series in a window
   - Average /median value in a window
   - Window variance
 - Date and time features
   - Minute of an hour, hour of a day, day of week, etc
   - Tag dates as holidays,etc.
 - Target encoding
 - Forecasts from other models

### Lags of the time series

Lets say we add the lag of the target variable from 6 steps back up to 24 - Need to look
at the data which was 6 steps before. Basically just a sliding window that we put into our vector.


### Reguarlization

Not all features are equally healthy - you don't want all the features to be highly correlated with
each other.

We use Lasso / Ridge Regression.

Ridge Regression just adds the L2 norm, which penalizes the coefficients of the model.

In our example LASSO removed more features that were highly correlated and not relevant
for explaining the data.

## Tree Based Models

Using tree-based models in data that has nay kind of trend is not a good idea. We don't want
the model to under-predict in the case of a linear trend. The data needs to be stationary.

## Facebook Prophet (fbprohphet)

Open source tool for timeseries. Allows us to use the forecasting instantly without the need
for any transformations.

You need to transform your time series into a two-column data frame (ds and y). Fitting happens
lazily.

In the model there's a seasonality flag, you need to provide it with the ones that you really
want to have.

We now have all kinds of components - uses the same decomposition of the initial time series
(trend, weekly seasonality, etc)
