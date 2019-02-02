from sklearn.preprocessing import StandardScaler

from utils.dataframe import drop_columns_from_dataframes
from utils.model import (
    expand_onehot_encoding,
    rescale_non_categorical_data
)

def featurize_for_all_models(drop_columns, *dataframes):
    return None, drop_columns_from_dataframes(drop_columns, *dataframes)


def featurize_for_tabular_models(drop_columns, categorical_features):
    def inner(*dataframes):
        data_info, dataframes = featurize_for_all_models(drop_columns, *dataframes)
        dataframes = drop_columns_from_dataframes(['listing_id', 'label_interest_level'], *dataframes)

        # Need to rescale data here
        return data_info, tuple(
            expand_onehot_encoding(rescale_non_categorical_data(dataframe, categorical_features), categorical_features)
            for dataframe in dataframes
        )

    return inner


def featurize_for_tree_models(drop_columns, categorical_features):
    def inner(*dataframes):
        data_info, dataframes = featurize_for_all_models(drop_columns, *dataframes)
        dataframes = drop_columns_from_dataframes(['listing_id', 'label_interest_level'], *dataframes)

        # Only difference is that we don't rescale the data
        return data_info, tuple(
            expand_onehot_encoding(dataframe, categorical_features)
            for dataframe in dataframes
        )

    return inner

