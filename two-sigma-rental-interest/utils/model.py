"""/utils/model.py

Models to use with the data.

This module creates pipelines, which depending on the underlying
model, will one-hot encode categorical data or just leave it as is,
converting it to a number. All the returned models satisfy the
sklearn estimator API, so we can use them with grid search/evolutionary
algorithms for hyperparameter search if we want to.
"""

import itertools
import numpy as np
import pandas as pd
import xgboost as xgb

from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_squared_error,
    log_loss
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.report import generate_classification_report_from_preds


def expand_onehot_encoding(dataframe, categorical_columns):
    for column, dimension in categorical_columns.items():
        encoding_mat = OneHotEncoder(categories=[range(dimension)]).fit_transform(dataframe[column].values.reshape(-1, 1)).todense()
        encoded = pd.DataFrame(encoding_mat,
                               columns=['{}_{}'.format(column, i) for i in range(dimension)])
        dataframe = pd.concat((
            dataframe.drop(column, axis=1),
            encoded
        ), axis=1)

    return dataframe


def sklearn_pipeline_steps(categorical_columns, verbose=False):
    return [
        ('scaling', StandardScaler())
    ]


def basic_logistic_regression_pipeline(categorical_columns,
                                       verbose=False):
    return LogisticRegression(multi_class='multinomial', solver='newton-cg')


def basic_random_forest_pipeline(featurized_train_data,
                                 train_labels,
                                 verbose=False,
                                 param_grid_optimal=None,
                                 **kwargs):
    pipeline = RandomForestClassifier(
        random_state=42,
        **kwargs
    )

    # Grid search if we don't get an optimal parameter grid
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [3, 5],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [100, 150, 200, 250]
    }
    search = GridSearchCV(pipeline,
                          param_grid_optimal or param_grid,
                          cv=StratifiedKFold(n_splits=2, shuffle=True).split(featurized_train_data,
                                                                             train_labels),
                          refit=True,
                          verbose=50,
                          n_jobs=8,
                          scoring=make_scorer(log_loss,
                                              greater_is_better=False,
                                              needs_proba=True))
    return search


def basic_xgboost_pipeline(featurized_train_data,
                           train_labels,
                           verbose=False,
                           train_param_grid_optimal=None,
                           **kwargs):
    pipeline = xgb.XGBClassifier(
        seed=42,
        objective='multi:softprob',
        **kwargs
    )

    # Grid search if we don't get an optimal parameter grid
    param_grid = {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5],
        'n_estimators': [100, 150, 200]
    }
    search = GridSearchCV(pipeline,
                          train_param_grid_optimal or param_grid,
                          cv=StratifiedKFold(n_splits=2, shuffle=True).split(featurized_train_data,
                                                                             train_labels),
                          refit=True,
                          verbose=50,
                          n_jobs=8,
                          scoring=make_scorer(log_loss,
                                              greater_is_better=False,
                                              needs_proba=True))
    return search


def basic_extratrees_pipeline(featurized_train_data,
                              train_labels,
                              verbose=False,
                              param_grid_optimal=None,
                              **kwargs):
    pipeline = ExtraTreesClassifier(
        random_state=42,
        **kwargs
    )

    # Grid search if we don't get an optimal parameter grid
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [3, 5],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [100, 150, 200, 250]
    }
    search = GridSearchCV(pipeline,
                          param_grid_optimal or param_grid,
                          cv=StratifiedKFold(n_splits=2, shuffle=True).split(featurized_train_data,
                                                                             train_labels),
                          refit=True,
                          verbose=50,
                          n_jobs=8,
                          scoring=make_scorer(log_loss,
                                              greater_is_better=False,
                                              needs_proba=True))
    return search


def basic_adaboost_pipeline(featurized_train_data,
                            train_labels,
                            verbose=False,
                            param_grid_optimal=None,
                            **kwargs):
    pipeline = AdaBoostClassifier(
        random_state=42,
        **kwargs
    )

    # Grid search if we don't get an optimal parameter grid
    param_grid = {
        'learning_rate': [10e-2, 10e-1, 1, 2, 5],
        'n_estimators': [100, 150, 200, 250]
    }
    search = GridSearchCV(pipeline,
                          param_grid_optimal or param_grid,
                          cv=StratifiedKFold(n_splits=2, shuffle=True).split(featurized_train_data,
                                                                             train_labels),
                          refit=True,
                          verbose=50,
                          n_jobs=8,
                          scoring=make_scorer(log_loss,
                                              greater_is_better=False,
                                              needs_proba=True))
    return search


def basic_svc_pipeline(featurized_train_data,
                       train_labels,
                       verbose=False,
                       param_grid_optimal=None,
                       **kwargs):
    pipeline = SVC(
        random_state=42,
        probability=True,
        **kwargs
    )

    # Grid search if we don't get an optimal parameter grid
    param_grid = {
        'C': [0.1, 1.0],
        'gamma': ['auto', 'scale'],
        'kernel': ['linear', 'rbf', 'poly'],
    }
    search = GridSearchCV(pipeline,
                          param_grid_optimal or param_grid,
                          cv=StratifiedKFold(n_splits=2, shuffle=True).split(featurized_train_data,
                                                                             train_labels),
                          refit=True,
                          verbose=50,
                          n_jobs=8,
                          scoring=make_scorer(log_loss,
                                              greater_is_better=False,
                                              needs_proba=True))
    return search


def calculate_statistics(statistics, test_labels, predictions):
    return {
        k: s(test_labels, predictions)
        for k, s in statistics.items()
    }


def format_statistics(calculated_statistics):
    return ", ".join([
        "{0}: {1:.2f}".format(k, s)
        for k, s in calculated_statistics.items()
    ])


def prediction_accuracy(labels, predictions):
    return accuracy_score(labels, np.argmax(predictions, axis=1))


def fit_one_split(model, features, labels, statistics, train_index, test_index):
    train_data, train_labels = features.iloc[train_index], labels[train_index]
    test_data, test_labels = features.iloc[test_index], labels[test_index]

    model.fit(train_data, train_labels)
    predictions = model.predict_proba(pd.DataFrame(test_data, columns=features.columns))

    return (
        test_labels,
        predictions,
        calculate_statistics(statistics,
                             test_labels.values,
                             predictions)
    )


def test_model_with_k_fold_cross_validation(model,
                                            features,
                                            labels,
                                            statistics,
                                            n_splits=5,
                                            random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_labels, predictions = [], []

    for i, (train_index, test_index) in enumerate(kf.split(features, labels)):
        fold_test_labels, fold_predictions, calculated_statistics = fit_one_split(
            model,
            features,
            labels,
            statistics,
            train_index,
            test_index
        )
        print('Fold', i, format_statistics(calculated_statistics))
        test_labels.extend(fold_test_labels)
        predictions.extend(fold_predictions)

    return (
        calculate_statistics(statistics, test_labels, predictions),
        test_labels,
        predictions,
        model
    )


def get_prediction_probabilities_with_columns_from_predictions(listing_ids,
                                                               predictions):
    return pd.concat((pd.DataFrame(listing_ids, columns=['listing_id']),
                      pd.DataFrame(predictions, columns=('high', 'medium', 'low'))),
                     axis=1)


def get_prediction_probabilities_with_columns(model, listing_ids, test_data):
    predictions = model.predict_proba(test_data)
    return get_prediction_probabilities_with_columns_from_predictions(listing_ids,
                                                                      predictions)


def write_predictions_table_to_csv(predictions_table, csv_path):
    predictions_table.to_csv(csv_path, columns=['listing_id', 'high', 'medium', 'low'], index=False)


def split_into_continuous_and_categorical(categorical_columns, *dataframes):
    return tuple([
        (df.drop(['listing_id', 'label_interest_level'] + list(categorical_columns.keys()), axis=1).values,
         df[list(categorical_columns.keys())].values)
        for df in dataframes
    ])


def check_if_any_nan(rescaled_continuous_data):
    """Throws an exception if any components of a datapoint are NaN."""
    if rescaled_continuous_data[np.isnan(rescaled_continuous_data)].size != 0:
        raise RuntimeError("""Some elements of array are NaN.""")


def rescale_non_categorical_data(dataframe, categorical_features):
    features_columns = [c for c in dataframe.columns if c.startswith("features_")]
    features_data = dataframe[features_columns].reset_index(drop=True)
    categorical_data = dataframe[list(categorical_features.keys())].reset_index(drop=True)
    continuous_data = dataframe.drop(list(categorical_features.keys()) + features_columns, axis=1)

    # Now, rescale the values and create a new dataframe
    rescaled_continuous_data = StandardScaler().fit_transform(continuous_data.values)
    check_if_any_nan(rescaled_continuous_data)

    return pd.DataFrame(np.hstack((rescaled_continuous_data, categorical_data.values, features_data.values)),
                        columns=list(continuous_data.columns) + list(categorical_data.columns) + features_columns)


def rescale_features_and_split_into_continuous_and_categorical(categorical_columns,
                                                               *dataframes):
    rescaled_dataframes = tuple([rescale_non_categorical_data(df, categorical_columns) for df in dataframes])
    return split_into_continuous_and_categorical(categorical_columns, *rescaled_dataframes)


def train_model_and_get_validation_and_test_set_predictions(
    train_dataframe,
    validation_dataframe,
    test_dataframe,
    raw_validation_dataframe,
    train_labels,
    validation_labels,
    featurizer,
    model_training_func,
    model_prediction_func,
    train_param_grid_optimal=None
):
    (data_info,
     (featurized_train_data,
      featurized_validation_data,
      featurized_test_data)) = featurizer(train_dataframe,
                                          validation_dataframe,
                                          test_dataframe)

    model = model_training_func(data_info,
                                featurized_train_data,
                                featurized_validation_data,
                                train_labels,
                                validation_labels,
                                train_param_grid_optimal=train_param_grid_optimal)
    validation_preds, validation_probabilities = model_prediction_func(
        model,
        featurized_validation_data
    )
    report = generate_classification_report_from_preds(validation_preds,
                                                       validation_probabilities,
                                                       raw_validation_dataframe,
                                                       validation_labels,
                                                       [0, 1, 2],
                                                       columns=['description'])
    display(report)

    test_preds, test_probabilities = model_prediction_func(
        model,
        featurized_test_data
    )

    return validation_probabilities, test_probabilities


def predict_with_sklearn_estimator(model, data):
    return model.predict(data), model.predict_proba(data)


