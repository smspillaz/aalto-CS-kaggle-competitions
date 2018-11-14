"""/data.py

Tools for loading data.
"""

import errno
import itertools
import json
import os
import pandas as pd


def load_json_from_path(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise error

    return None


def json_to_pandas_dataframe(dictionary):
    columns = list(dictionary.keys())
    rows = sorted(list(set(itertools.chain.from_iterable([
        list(dictionary[k].keys())
        for k in columns
    ]))), key=lambda x: int(x))

    # map(list, zip(*data)) is a quick trick to transpose
    # a list of lists
    data = list(map(list, zip(*([rows] + [
        [
            dictionary[column][r] if r in dictionary[column] else None
            for r in rows
        ]
        for column in columns
    ]))))
    df = pd.DataFrame(data, columns=['id'] + columns)
    df.set_index('id')

    return df


def load_training_test_data(training_data_path, test_data_path):
    return (
        json_to_pandas_dataframe(load_json_from_path(training_data_path)),
        json_to_pandas_dataframe(load_json_from_path(test_data_path))
    )
