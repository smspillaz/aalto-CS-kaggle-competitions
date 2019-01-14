import datetime
import itertools
import json
import operator
import os
import pandas as pd
import pprint
import numpy as np
import re
import spacy
import torch

from collections import Counter, deque
from sklearn.metrics import mean_squared_error

from data import load_training_test_data
from dataframe import (
    categories_from_column,
    column_list_to_category_flags,
    count_ngrams_up_to_n,
    map_categorical_column_to_category_ids,
    normalize_categories,
    remap_column,
    remap_date_column_to_days_before,
    remove_small_or_stopwords_from_ranking
)
from doc2vec import (
    column_to_doc_vectors
)
from model import (
    basic_logistic_regression_pipeline,
    format_statistics,
    get_prediction_probabilities_with_columns,
    prediction_accuracy,
    test_model_with_k_fold_cross_validation
)

#nlp = spacy.load("en")

TRAIN_DF, TEST_DF = load_training_test_data(os.path.join('data', 'train.json'),
                            os.path.join('data', 'test.json'))

import fastai.text.data
from fastai.text import *
from sklearn.model_selection import train_test_split

#train_lm_df, valid_lm_df = train_test_split(pd.concat([TRAIN_DF,
#                                                       TEST_DF]))

bs=48

# works around a bug in fast.ai
tokenizer = Tokenizer()
tokenized_texts = tokenizer.process_all(pd.concat([TRAIN_DF, TEST_DF])["description"])
vocab = Vocab.create(tokenized_texts, max_vocab=60000, min_freq=2)

data_lm = (TextList.from_df(pd.concat([TRAIN_DF,
                                       TEST_DF]),
                            path='renthop_lm',
                            cols=['description'],
                            vocab=vocab)
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .random_split_by_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))
data_lm.save('tmp_lm')

data_lm = TextLMDataBunch.load('data_lm', 'tmp_lm', bs=bs)

import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

learn = language_model_learner(data_lm,
                               pretrained_model=URLs.WT103_1,
                               drop_mult=0.3)

# Found that 1e-1 worked well here
learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))

learn.save('fit_head_renthop')
print('Loading head')
learn.load('fit_head_renthop')

learn.unfreeze()

# Drop by an order of magnitude when fine-tuning the rest
# layers
learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))

print('Saving encoder')
learn.save_encoder('fine_tuned_renthop')
