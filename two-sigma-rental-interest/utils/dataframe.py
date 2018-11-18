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


def remap_to_float(data_frame, column, new_column, mapping):
    data_frame[new_column] = data_frame[column].transform(lambda x: mapping[x])
    return data_frame


def remap_date_column_to_days_before(data_frame,
                                     column,
                                     new_column,
                                     reference_date):
    data_frame[new_column] = data_frame[column].transform(
        lambda x: (reference_date - datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).days
    )
    return data_frame
