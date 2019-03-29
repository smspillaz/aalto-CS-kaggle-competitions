# Ensembling Techniques

You have many classifiers which make a "melody" out of the data.

"Use shitty algorithms, with shitty data and shitty computers"

## Simple Ensembles
 - Committees: Different classifiers, each of them vote (nobody uses this)
 - Weighted Averages: "Up-weight" the better predictors

Variance and Bias:
 - Error emerging from any model can be broken down into three components
   mathematically:
   - Error(x) = Bias^2 + Variance + Irreducible Error

A high bias error means that you have under-performing model.

A high variance model is an overfitting model on your training population.

You can't really have low variance and low bias. A champion model balances
between variance and bias.

## Bootstrapping

Sort of like random sampling with replacement.

## Bagging

This is a simple and powerful ensembling method.

It is the application of the Bootstrap procedure to a high variance
machine learning algorithm, typically decision trees.

So you have different test sets and different classifiers applied to each.

Thsi can handle higher dimensionality of the data quite well (random forests)
and maintains accuracy for missing data.

### Bagging with respect to classification

Generate samples with your bootstrapping.

The final classifier will average the outputs from all these individual classifiers.

The final classifier just average the outputs.

The errors kind of cancel each other out.

### Bagging with regression

How does bagging reduce your variance?

The mean error over all the reression functions is just 1/n

## Random Forest Ensembling

We take a random selection of features rather than using all the feautres to grow
trees.

## Boosting

This is something that is sequential as opposed to parallel. This converts
weak learners into strong learners.

We train the weak learners sequentially by correcting each weak learner until
it gets better.

### AdaBoost

You input your data, its classified, you find out how good that classifier was,
then you try and improve the "wrong" classifications. You find a classifier which has a
very low-weighted error.

Choose your classifier weight based on the error that you get. Choose classifiers
that have low error and after you do that, force the model to concentrate on the
datapoints which were misclassified more.

So, first you calculate the error:

$\text{MME_{\text{emp}}} = \frac{\sum^N w_i I(y_i \ne h_j(x_i))}{\sum^N w_i}$

Basically the lower your error rate, the higher the weight that you give to
that classifier.

You're changing the penalty on each training example based on how badly it was
misclassified by the last classifier.

### Gradient Boosting

You have your error residuals for each data point (classifier) and the idea is to bring down
the gradient of the error residuals.

In this case you change the penalty on each training example based on the gradient
of the residuals.

Problem: This overfits! That's what you have to deal with - check to see if you're issue
is a bias issue or a variance issue. Set an appropriate number of trees in your model, train
your model and check your score on the validation dataset - then increase the number of trees -
if the error increases, then don't add more trees.

That said, it may be better to have one validation set for early stopping, then one for
your actual validation data.

You can also use L1/L2 regularization - if you see a huge difference between score on
your training data and validation data (its probably because you have a huge bias), so
you need more regularization.

Play with toy datasets.

### XGBoost

CatBoost performs really well on categorical data. This is very slow though, because it
calculates the gradient for each data point independently.

LightGBM is 3-5x faster than CatBoost

### Stacking
Form a matrix on the predictions, then train another model to classify from
the original predictions.


#### Stacking different folds
You can divide your data into folds, take one fold as your holdout validation data,
take another fold (n) as your output predictions for each "fold model" and then use the
rest of the fold data to train.

Then stack all the predictions on each (n) fold together and train a model on *those* predictions.


