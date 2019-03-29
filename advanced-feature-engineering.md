# Advanced feature engineering
Everything is a hyperparameter

Two approaches:
 - Feature engineering on domain understanding
 - Generate lots of features that seem useless.

Several approaches for processing:
 - fill NA with median/0/mean
 - Categorical columns, label encoding / categorical columns -> ohe, categorical columns -> target enc / nn / lightgm / logreg

Need to understand what type of hyperparameters are good to test and what are not.

Where to find inspiration:
- Read papers (Google Scholar, Arxiv, Books)
- Previous winners of competitions (http://ndres.me/kaggle-past-solutions)

## Numerical Features:
 - Basic operations (nf1 + nf2, nf1 - nf2 etc)
 - Works well for tree based models, but not for NNs because the linear layers will do that for you.

## Encoding
 - When you have categorical data, you can encode the categories with some statistics of
   numerical features. You could encode with mean value and calculate the distances between
   the mean value and the current value (eg, L1, L2, cosine distance, etc)
 - Group all the numerical features into category groups, calculate statistics for each one,
   then add new cells to calculate the distance between your numerical feature to the statistic
   numerical feature for that group.
   - For instance, average sales on a sunday, compared to sales that day.
   - Makes sense to compare within groups, not between groups

## Time series:
 - Rolling statistics over different windows (mean, min, max, std, median, max-min, quantiles)
 - Compute detla with the numerical feature and the rolling statistic
 - Compare derivative (speed of change)
 - tsfresh

Good ideas to normalize the timeseries data, such that two timeseries which look the same
have a zero-mean and unit variance.

## Categorical Features
 - One-hot encoding
 - Current SOTA: Frequency encoding / Target encodings
   - Frequency Encoding: Encode your feature with number of times that this feature was represented in your dataset.
   - Target Encoding: Encode with some statistic about a numerical feature in your dataset (mean of the target for
     the given category) (better for tree-based methods)
   - Embeddings (much better for NNs) [merges similar categories in encoding space]
 - This helps to deal with a lot of categories.

### How do you train the embeddings?
Need to train them end to end and your target is the thing that you're trying to predict. Or
you could predict other features?

What is the difference between this and PCA?
 - Usually its not a good idea to use PCA with NNs
   since your data is not typically linearly dependent.

## Mean-encoding
 - It makes sense to split your data several wans and then do your mean encoding
 - Then take the average of the test predictions as your encoding for the mean.
 - Just fitting a linear model on a one-hot representation of your dataset.
 - Separate linear model for each column

## Intersection of categorical features
 - Combine categorical features (concatenating the labels into "new" categories)
 - Good to find outliers or small intersections

## Time features
 - Time features can be turned into categories (eg, month of year, week of year,
   day of week, quarter, is_holiday, etc)
 - Time between events (the time after the last user action, time between actions)

## Geographical Features
 - OpenStreetMap (find nearby objects, distance to an ATM for instance for sales)
 - Satellite Images - use as additional photo data
 - Clustering of co-ordinates (kmeans, dbscan)

## Text features
 - Count vectorizers, tf-idf
 - Meta-features (prediction of LR or SVM as new feature)
 - word2vec, sentence2vec

## Pictures
 - Output of some neural network and use it as an additional feature (basically
   encoding the image as a n-length feature)

## Bias-Variance Tradeoff
 - More features -> less bias, more variance
 - Adjust your model by adding regularization
 - Combat variance by having lots of models
 - LightGBM
