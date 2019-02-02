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
    return Pipeline((
        *sklearn_pipeline_steps(categorical_columns, verbose=verbose),
        ('logistic', LogisticRegression(multi_class='multinomial',
                                        solver='newton-cg'))
    ))


def basic_xgboost_pipeline(categorical_columns,
                           verbose=False,
                           n_estimators=1000,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           **kwargs):
    return Pipeline([
        ('xgb', xgb.XGBClassifier(
            n_estimators=n_estimators,
            seed=42,
            objective='multi:softprob',
            subsample=0.8,
            colsample_bytree=0.8,
            **kwargs
        ))
    ])


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


def rescale_features_and_split_into_continuous_and_categorical(categorical_columns,
                                                               *dataframes):
    split_dfs = split_into_continuous_and_categorical(categorical_columns, *dataframes)
    scaler = StandardScaler()

    return tuple([
        (scaler.fit_transform(continuous), categorical)
        for continuous, categorical in split_dfs
    ])

