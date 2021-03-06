"""/utils/dataframe.py

Utilities to clean out the data
in the dataframe.
"""

import datetime
import functools
import itertools
import json
import numpy as np
import operator
import pandas as pd
import pprint
import re
import spacy

from collections import Counter, deque
from imblearn.over_sampling import RandomOverSampler


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
        lambda x: (reference_date - datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")).days
    )
    return data_frame


def map_categorical_column_to_category_ids(column,
                                           new_column,
                                           *dataframes,
                                           min_freq=1):
    categories = list(itertools.chain.from_iterable([
        df[column] for df in dataframes
    ]))
    category_counts = Counter(categories)
    category_to_unknown_mapping = {
        category: category if count >= min_freq else "Unknown"
        for category, count in category_counts.items()
    }
    categories_set = set([category_to_unknown_mapping[c] for c in categories])
    category_to_id_map = {
        category: i
        for i, category in enumerate(sorted(categories_set))
    }
    id_to_category_map = {
        i: category
        for category, i in category_to_id_map.items()
    }

    return ((
        category_to_unknown_mapping,
        category_to_id_map,
        id_to_category_map
    ), (
        remap_column(df,
                     column,
                     new_column,
                     lambda x: category_to_id_map[category_to_unknown_mapping[x]])
        for df in dataframes
    ))


def remap_columns_with_transform(column,
                                 new_column,
                                 transform,
                                 *dataframes):
    """Remove some columns with a transform."""
    return (
        remap_column(df,
                     column,
                     new_column,
                     transform)
        for df in dataframes
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

    return dataframe


def numerical_feature_engineering(numerical_columns, *dataframes):
    """Add, subtract, divide, multiply, exponentiate and take log."""
    return (
        numerical_feature_engineering_on_dataframe(df,
                                                   numerical_columns)
        for df in dataframes
    )


def normalize_eastwest(eastwest):
    eastwest = eastwest.lower().strip()

    if not eastwest:
        return ""

    if eastwest[0] == "e":
        return "e"
    elif eastwest[0] == "w":
        return "w"
    else:
        return ""


STREET_MAPPING = {
    "st": "street",
    "ave": "avenue",
    "pl": "place",
    "rd": "road"
}


def normalize_name(name):
    m = re.match(r"(?P<address>[\w\s]+)(?P<st>st|street|ave|avenue|place|pl|road|rd).*",
                 name.lower().strip())

    if not m:
        return name.lower().strip()

    return "{address} {street}".format(
        address=m.groupdict()["address"].strip(),
        street=STREET_MAPPING.get(m.groupdict()["st"], m.groupdict()["st"])
    )


def normalize_address(address_dict):
    return "{eastwest} {name}".format(
        eastwest=normalize_eastwest(address_dict["eastwest"] or ""),
        name=normalize_name(address_dict["name"] or "")
    )


def parse_address_components_from_address(address):
    m = re.match(r"(?P<number>[0-9]*\s+)?\s*(?P<eastwest>East|West|E\s|W\s)?\s*(?P<name>[A-Za-z0-9\.\-\s]*)",
                 normalize_whitespace(address),
                 flags=re.IGNORECASE)
    return {
        "normalized": normalize_address(m.groupdict()) if m is not None else address
    }


def parse_address_components_for_column(dataframe, column):
    return pd.concat((dataframe, pd.DataFrame.from_records([
        {
            "{}_{}".format(column, key): value for key, value in
            parse_address_components_from_address(cell).items()
        }
        for cell in dataframe[column]
    ])), axis=1)


def parse_address_components(columns, *dataframes):
    return (
        functools.reduce(lambda df, c: parse_address_components_for_column(df,
                                                                           c),
                         columns,
                         df)
        for df in dataframes
    )


def count_json(json_data):
    return len(json_data)


def count_json_in_dataframes(column, *dataframes):
    return remap_columns_with_transform(column,
                                        "{}_count".format(column),
                                        count_json,
                                        *dataframes)


def random_oversample_dataframe(dataframe):
    """Oversample the dataframe using RandomOverSampler.

    This works on non-numerical data but doesn't do any synethetic
    data generation.
    """
    return pd.DataFrame(
        RandomOverSampler().fit_resample(dataframe, dataframe["label_interest_level"])[0],
        columns=dataframe.columns
    )


def drop_column_if_present(drop_columns, dataframe):
    columns = set(dataframe.columns)
    return dataframe.drop([d for d in drop_columns if d in columns], axis=1)


def drop_columns_from_dataframes(drop_columns, *dataframes):
    return (
        drop_column_if_present(drop_columns, df)
        for df in dataframes
    )


def remove_outliers(dataframe, column_quantile_dict):
    """Remove anything with price above a certain quantile."""
    for column, (quantile_min, quantile_max) in column_quantile_dict.items():
        q_max = dataframe[column].quantile(quantile_max)
        q_min = dataframe[column].quantile(quantile_min)
        dataframe = dataframe[dataframe[column] < q_max]
        dataframe = dataframe[dataframe[column] > q_min]
    return dataframe.reset_index(drop=True)

