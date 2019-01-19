"""/utils/model.py

Models to use with the data.

This module creates pipelines, which depending on the underlying
model, will one-hot encode categorical data or just leave it as is,
converting it to a number. All the returned models satisfy the
sklearn estimator API, so we can use them with grid search/evolutionary
algorithms for hyperparameter search if we want to.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler


def expand_onehot_encoding(dataframe, categorical_columns):
    for column in categorical_columns:
        encoding_mat = OneHotEncoder().fit_transform(dataframe[column].values.reshape(-1, 1)).todense()
        encoded = pd.DataFrame(encoding_mat,
                               columns=['{}_{}'.format(column, i) for i in range(max(dataframe[column].values) + 1)])
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


def get_prediction_probabilities_with_columns(model,
                                              test_dataframe,
                                              keep_columns):
    return pd.concat((test_dataframe[keep_columns],
                      pd.DataFrame(model.predict_proba(test_dataframe.drop(keep_columns, axis=1)))),
                     axis=1)


def rescale_features_and_split_into_continuous_and_categorical(features_train_dataframe,
                                                               features_test_dataframe,
                                                               categorical_columns):
    return (
        StandardScaler().fit_transform(features_train_dataframe.drop(['listing_id',
                                                                      'label_interest_level'] + categorical_columns,
                                                                     axis=1)),
        features_train_dataframe[categorical_columns],
        StandardScaler().fit_transform(features_test_dataframe.drop(['listing_id',
                                                                     'label_interest_level'] + categorical_columns,
                                                                     axis=1)),
        features_test_dataframe[categorical_columns]
    )
