# Feature Engineering

One-hot encdoing: Each class value is encoded in a binary manner, but where does it work?
 - Does not have any effect on decision trees, since each class can be encoded to a numer and
   cut on specific area
 - But for linear models, it preserves the information that the numbers are distinct.
 - If linear dependence ofthe feature does not matter, then a tree based appraoch could be used - where
   you just divide the surface and don't care about linear dependence.


MinMax vs Scandard Scaling:
 - Efficeint for gradient descent models (scaled from zero to one)
 - Standard Scaling transforms the feature to have zero mean and unit variance. This is useful
   when your data scalesa are all over the place (sometimes its better to scale by some robust
   scaling which does not take into account outliers, such as median)
 - QuantileNormalizer
 - You can take the first to 99th percentiles before you do min/max scaling - this likely eliminates
   outliers. Or you can just keep on decreasing your range until the variance derivative drops
   sharply.

Feature Combinations:
 - For instance, combining multiple features that give you distance in metres
   such that you can get total distance.
 - Or for instance, adding ratios (distance per headshot in pubg competition).
 - This is stuff that the model might not work out on its own but is highly relevant.
 - Sometimes it makes sense to just do arithmetic on all your continuous variables
   because their combinations could be highly relevant.
 - If there are groups of rows, it makes sense to take summary statistics (mean, max,
   median) of each team.
 - You can check how good an arithmetic feature is by just doing linear regression
   *only* with that feature.

Date Features:
 - These can be transformed into seconds-from-n or days-from-n. Depends on your problem.
 - Also add flags ("is on weekend", "is on working hours") etc.

Showing the most important features:
 - Use LGBMRegressor - to show the most important / predictive features
 - sklearn's RFECV can be used to choose the best n features for the model.
