# GBM

Decision Trees.

GBM grows the tree level-wise as opposed to leaf-wise.

Most of them are pretty easy to use, just need to turn the hyperparameters:
 - Grid search (takes a very long time, guaranteed to converge within domain)
 - Random search (guarnatted to converge but takes a very very long time, possibly infinite)
 - Bayesian Optimization or Evolutionary Algorithms (local optima, but good at finding them).

Hyperparameter Optimization Software
 - Hyperopt
 - Sciki-Optimize
 - GPyOpt
 - GoBo
 - SMAC3

Try to understand if your model is overfitting or underfitting.

If you use a large number of leaves or a high depth this may cause overfitting.

`min_data_in_leaf`: Setting it to a large value can avoid growing too deep a tree but may
cause under-fitting. Setting it to hundreds or thousands is enough for a large dataset.

More parameters:
 - num leaves
 - learningrate:
 - feature_fraction: If hte model is overfitting, lower
 - bagging_fraction: specifies the fraction of data to be used for each iteration
 - min_child_samples: minimal number of data in one leaf
 - min_split_gain: minimal gain to perform split
 - reg_lambda: L1 regularization
 - reg_lambda: L2 regularization

Probably want to set the early stopping threshold too to avoid overfitting.

## Mathematical Intuition
It just makes a bunch of decision trees. Eg, if feature X > Y, take this branch,
or take another branch.

In RF we take the average of the predictions

Gradient Boosting
 - At each iteration calculate our prediction of previous trees and compare it
   to the actual values. Then we fid the next tree based on these differences.
 - By summing up the trees you receive more accurate predictions than by just
   averaging them.
 - You can do it with weights of each tree = 1, but the variance might change.
 - Each tree wants to predict the target variable as accurately as possible, combine
   them.
