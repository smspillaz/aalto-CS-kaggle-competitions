import numpy as np
import torch
from random import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def train_test_split_indices_maintain_relative_ordering(indices, probability_test):
    """Do a train/test split which distributes indices into two sets.

    The expected length of the test set is probability_test * len(indices), it
    won't be exactly this but that doesn't matter.

    We go through the indices and allocate them one by one to either the
    training set or the test set.
    """
    training_set = []
    test_set = []

    for idx in indices:
        if random() < probability_test:
            test_set.append(idx)
        else:
            training_set.append(idx)

    return training_set, test_set


def simple_train_test_split_without_shuffle_func(proportion):
    """We need this override here to split without shuffling.

    We need to split without shuffling since packed RNN sequences require
    batches to be in descending order of variable lengths (presumably this
    is some sort of performance optimization in the CuRNN implementation).

    Note that this split is really only for tracking runtime statistics
    while the net is being trained, for actual cross validation it is
    better to set up a pipeline and do the splitting there, encoding
    all the sequences after the split.
    """
    def func(dataset, y=None, groups=None):
        idx_train, idx_test = train_test_split_indices_maintain_relative_ordering(list(range(0, len(dataset))),
                                                                                  proportion)

        return (
            torch.utils.data.Subset(dataset, idx_train),
            torch.utils.data.Subset(dataset, idx_test),
        )

    return func


def tensor_to_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu()

    return tensor


def ordered_train_test_split_with_oversampling(proportion):
    """Do the train test split, but oversample the training set and reorder it."""
    def func(dataset, y, groups=None):
        idx_train, idx_test = train_test_split_indices_maintain_relative_ordering(list(range(0, len(dataset))),
                                                                                  proportion)

        idx_train, labels = RandomOverSampler().fit_resample(np.array(idx_train).reshape(-1, 1),
                                                             [int(y[i]) for i in idx_train])
        idx_train = sorted(idx_train.flatten())

        return (
            torch.utils.data.Subset(dataset, idx_train),
            torch.utils.data.Subset(dataset, idx_test)
        )

    return func
