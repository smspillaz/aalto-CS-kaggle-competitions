#!/usr/bin/env python
#
# Generate the doc2vec embeddings in a separate
# callable process.

import argparse
import numpy as np
import pandas as pd

from doc2vec import documents_to_vectors_model
from data import load_training_test_data
from dataframe import (
    normalize_description,
    remap_column,
    remap_columns_with_transform
)


def main():
    """Entry point for the module."""
    parser = argparse.ArgumentParser("""generate-doc2vec.py""")
    parser.add_argument("train_data",
                        type=str,
                        help="The training data")
    parser.add_argument("test_data",
                        type=str,
                        help="The testing data")
    parser.add_argument("output",
                        type=str,
                        help="Where to write the doc2vec weights")
    parser.add_argument("--batch-size",
                        type=int,
                        default=100,
                        help="What batch size to use when training.")
    parser.add_argument("--sentence-length",
                        type=int,
                        default=1000,
                        help="How long each sentence within a batch can be")
    parser.add_argument("--epochs",
                        type=int,
                        default=300,
                        help="How many epochs to train to")
    parser.add_argument("--parameters",
                        type=int,
                        default=100,
                        help="How many parameters the sentence vectors have")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="The learning rate")
    args = parser.parse_args()

    train_dataframe, test_dataframe = load_training_test_data(
        args.train_data,
        args.test_data
    )

    train_dataframe = remap_column(train_dataframe, "interest_level", "label_interest_level", lambda x: {
        "high": 0,
        "medium": 1,
        "low": 2
    }[x])
    # The TEST_DATAFRAME does not have an interest_level column, so we
    # instead add it and replace it with all zeros
    test_dataframe["label_interest_level"] = 0

    train_dataframe, test_dataframe = remap_columns_with_transform(
        train_dataframe,
        test_dataframe,
        "description",
        "clean_description",
        normalize_description
    )

    train_descriptions = list(train_dataframe["clean_description"])
    test_descriptions = list(test_dataframe["clean_description"])
    labels = list(train_dataframe["label_interest_level"])

    documents_to_vectors_model(
        train_descriptions,
        test_descriptions,
        labels,
        args.epochs,
        args.parameters,
        args.learning_rate,
        save=args.output,
        sentence_length=args.sentence_length,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
