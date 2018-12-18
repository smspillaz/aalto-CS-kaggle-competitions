"""/utils/dataframe.py

Utilities to clean out the data
in the dataframe.
"""

import datetime
import itertools
import json
import numpy as np
import operator
import pandas as pd
import pprint
import re
import spacy

from collections import Counter, deque


def string_to_category_name(string):
    return string.lower().replace(" ", "_")


def categories_from_column(data_frame, column):
    return list(set(list(itertools.chain.from_iterable(
        data_frame[column].tolist()
    ))))


def normalize_whitespace(string):
    return re.sub(r"\s+", " ", string)


def normalize_category(category):
    return normalize_whitespace(re.sub(r"[\*\-\!\&]", " ", category.lower())).strip()


def normalize_categories(categories):
    return [
        normalize_category(c) for c in categories
    ]


def sliding_window(sequence, n):
    """Returns a sliding window of width n over data from sequence."""
    it = iter(sequence)
    window = deque((next(it, None) for _ in range(n)), maxlen=n)

    yield list(window)

    for element in it:
        window.append(element)
        yield list(window)


def create_ngrams(content, n):
    for ngram in sliding_window(content.split(), n):
        yield " ".join(ngram)


def create_ngrams_up_to_n(content, n):
    for i in range(n):
        yield from create_ngrams(content, i)


def count_ngrams_up_to_n(content, n):
    return Counter(list(create_ngrams_up_to_n(content, n)))


def remove_small_or_stopwords_from_ranking(ranking, nlp, min_len):
    for word, rank in ranking:
        if nlp.vocab[word].is_stop or len(word) < min_len:
            continue

        yield word, rank


def column_list_to_category_flags(data_frame, column, grams):
    categories = [
        "{}_{}".format(column, string_to_category_name(n))
        for n in grams
    ]
    row_cleaned_categories = [
        normalize_category(" ".join(r))
        for r in data_frame[column].tolist()
    ]
    category_flags = pd.DataFrame.from_records([
        [1 if gram in r else 0 for gram in grams]
        for r in row_cleaned_categories
    ], columns=categories)

    return pd.concat((data_frame, category_flags), axis=1)


def remap_column(data_frame, column, new_column, mapping):
    data_frame[new_column] = data_frame[column].transform(mapping)
    return data_frame


def remap_date_column_to_days_before(data_frame,
                                     column,
                                     new_column,
                                     reference_date):
    data_frame[new_column] = data_frame[column].transform(
        lambda x: (reference_date - datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).days
    )
    return data_frame


def map_categorical_column_to_category_ids(train_data_frame,
                                           test_data_frame,
                                           column,
                                           new_column,
                                           min_freq=1):
    category_counts = Counter(train_data_frame[column]) + Counter(test_data_frame[column])
    category_to_unknown_mapping = {
        category: category if count >= min_freq else "Unknown"
        for category, count in category_counts.items()
    }
    category_to_id_map = {
        category: i
        for i, category in enumerate(sorted([
            category_to_unknown_mapping[c] for c in
            (set(train_data_frame[column]) | set(test_data_frame[column]))
        ]))
    }
    id_to_category_map = {
        i: category
        for category, i in category_to_id_map.items()
    }

    return (
        category_to_unknown_mapping,
        category_to_id_map,
        id_to_category_map,
        remap_column(train_data_frame,
                     column,
                     new_column,
                     lambda x: category_to_id_map[category_to_unknown_mapping[x]]),
        remap_column(test_data_frame,
                     column,
                     new_column,
                     lambda x: category_to_id_map[category_to_unknown_mapping[x]])
    )


def remap_columns_with_transform(train_data_frame,
                                 test_data_frame,
                                 column,
                                 new_column,
                                 transform):
    """Remove some columns with a transform."""
    return (
        remap_column(train_data_frame,
                     column,
                     new_column,
                     transform),
        remap_column(test_data_frame,
                     column,
                     new_column,
                     transform)
    )


def normalize_description(description):
    """Normalize the description field."""
    description = description.lower()
    description = re.sub(r"<[^<]+?(>|$)", " ", description)
    description = re.sub(r"[0-9\-]+", " ", description)
    description = re.sub(r"[a-z0-9]@[a-z0-9]\.[a-z]", " ", description)
    description = re.sub(r"[\!]+", "! ", description)
    description = re.sub(r"[\-\:]", " ", description)
    description = re.sub("\*", " ", description)
    return re.sub(r"\s+", " ", description)


def add_epsilon(array):
    return np.array([a + 10e-10 if a == 0 else a for a in array])


def numerical_feature_engineering_on_dataframe(dataframe,
                                               numerical_columns):
    """Do per-dataframe feature engineering."""
    for lhs_column, rhs_column in itertools.combinations(numerical_columns, 2):
        dataframe['{}_add_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] + dataframe[rhs_column]
        dataframe['{}_sub_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] - dataframe[rhs_column]
        dataframe['{}_mul_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] * dataframe[rhs_column]
        dataframe['{}_div_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] / add_epsilon(dataframe[rhs_column])
        dataframe['{}_exp_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] ** dataframe[rhs_column]
        dataframe['{}_log_{}'.format(lhs_column, rhs_column)] = np.log(add_epsilon(dataframe[lhs_column].as_matrix().astype('double')),
                                                                       dataframe[rhs_column].as_matrix().astype('double'))

    return dataframe


def numerical_feature_engineering(train_data_frame,
                                  test_data_frame,
                                  numerical_columns):
    """Add, subtract, divide, multiply, exponentiate and take log."""
    return (
        numerical_feature_engineering_on_dataframe(train_data_frame,
                                                   numerical_columns),
        numerical_feature_engineering_on_dataframe(test_data_frame,
                                                   numerical_columns),
    )
