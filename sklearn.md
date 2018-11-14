# Overview of sklearn

We expect data to come in the form of an array (`np,array` or `scipy.sparse` matrix). 

Shape: `n_samples, n_features`. 

Artifical datasets: Quickly test if your model works on different toy datasets

Feature scaling: Some algorithms are sensitive to feature scaling (linear models) while others
are not (decision trees).

 - StandardScaler: Assume that each feature is normally distributed
 - RobustScaler: 
 - MinMaxScaler: From -1 to 1

Scaling happens on a feature basis.

Feature extraction: CountVectorizer: Computes the count of each word in the document. This
is useful when you want to measure the tf-idf in a document.

You can also extract n-grams, or lowercase, or preprocess text. (`sklearn.feature_extraction.text`). You
can also set a limit on your vocab or a minimum frequency to prevent your vocab from growing too large.

Feature Selection:

 - RFECV: Using cross-vaidation to find an optimal subset of features for your model.
 - SelectKBest:
 - SelectFromModel: 

Hyperparameter optimization:
 - `sklearn.grid_search.GridSearchCV`: `GridSearchCV(model. {...})`
 - `RidgeCV`

Dimensionality reduction:
 - `PCA`: Determinisitc way of mapping features to a lower dimensional space.
 - `TSNE`: Stochastic, transductive. Usually decrease to `d = 2`.
 - `SparseRandomProjection`: More memory efficient, faster.

Model Selection:
 - If you have no idea what you want to do, you can look at the
   scikitlearn algorithm cheat-sheet, there's a decision tree which
   tells you how to select a model.

Estimators:
 - Most important object in the sklearn API 
 - Every estimator implements a `.fit` function `(data, [targets])`
 - Every estimator implements a `.predict` function `(data)`
 - Also: `coeff_` (estimated model parameters)
   - `predict_proba`
 - `.transform`: All preprocessing models / PCA etc implement this.

Many other projects are compatible with sklearn API.

Meta-estimators: Estimators which wrap another estimator (pipelines,
bagging, ensembles).

Pipelines: You have multiple estimators, you can build them on top
of each other. You can pipe the output of PCA into SVM.

Ensembles: Take the aggregate of many estimators and ask them to vote,
take the majority vote, or weighted vote by probability, etc.
 - The more uncorrelated your estimators the better it will be.
 - `ensemble_clf = VotingClassifier(estimators=[('name', RandomForest())])`

Metrics:
 - `confusion_matrix`
 - `classification_report` (combines `f1_score` etc).

Splitting data:
 - Measuring error on the training set: `train_test_split`
 - Cross validation: Repeatedly split the data into train-test pairs
   - `cross_val_scrore(estimator, X, y, cv=5)`
   - You need to assume that each observation is independent.


TPOT:
 - Evolutionary algorithm to do model selection
 - Outputs a TPOT pipeline (a sklearn pipeline)

Performance:
 - n_jobs: Multiprocessing


