import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split


PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'

def maybe_cuda(tensor):
    """CUDAifies a tensor if possible."""
    if torch.cuda.is_available():
        return tensor.cuda()

    return tensor.cpu()


def maybe_cuda_all(*args):
    return tuple([maybe_cuda(a) for a in args])


def set_to_one_hot_lookups(items):
    item_to_one_hot = {
        str(c): i for i, c in enumerate(items)
    }
    one_hot_to_item = {
        i: str(c) for i, c in enumerate(items)
    }

    return item_to_one_hot, one_hot_to_item


def words_to_one_hot_lookups(nlp):
    def _inner(all_characters):
        nlp.max_length = len(all_characters) + 1
        with nlp.disable_pipes('tagger', 'ner', 'parser'):
            word_set = sorted(set([str(x) for x in nlp(all_characters)] + [
                PAD_TOKEN, EOS_TOKEN
            ]))

        return set_to_one_hot_lookups(word_set)
    return _inner


def chars_to_one_hot_lookups(all_characters):
    sorted_chars_set = sorted(set(all_characters) | set([PAD_TOKEN, EOS_TOKEN]))
    return set_to_one_hot_lookups(sorted_chars_set)


def to_words(seq):
    return [str(s) for s in seq]


def at_least_one_word(words, pad):
    return words if len(words) > 0 else [pad]


def descriptions_to_word_sequences(nlp):
    def _inner(descriptions, word_to_one_hot):
        with nlp.disable_pipes('tagger', 'ner', 'parser'):
            return [
                # Very important to cast w to str here, otherwise we try to look up
                # the token in the dict which won't work.
                [word_to_one_hot[str(w)] for w in to_words(nlp(s))] + [word_to_one_hot[EOS_TOKEN]]
                for s in descriptions
            ]

    return _inner

def descriptions_to_char_sequences(descriptions, char_to_one_hot):
    return [
        [char_to_one_hot[c] for c in s] + [char_to_one_hot[EOS_TOKEN]]
        for s in descriptions
    ]

def longest_sequence(sequences):
    return max([
        len(s) for s in sequences
    ])


def pad_sequences(sequences, max_len, pad_token):
    return [
        [
            t for t, i in itertools.zip_longest(s,
                                                range(max_len),
                                                fillvalue=pad_token)
        ]
        for s in sequences
    ], [len(s) for s in sequences]


def generate_description_sequences(train_descriptions,
                                   test_descriptions,
                                   dictionary_encoder,
                                   sequence_encoder):
    """Given some train_descriptions and test_descriptions, encode as padded tensors."""
    atom_to_one_hot, one_hot_to_atom = dictionary_encoder(' '.join(train_descriptions) +
                                                          ' '.join(test_descriptions))
    train_sequences = sequence_encoder(train_descriptions, atom_to_one_hot)
    test_sequences = sequence_encoder(test_descriptions, atom_to_one_hot)

    max_sequence_len = max([longest_sequence(train_sequences),
                            longest_sequence(test_sequences)])
    train_sequences, train_sequences_lengths = pad_sequences(train_sequences, max_sequence_len, atom_to_one_hot[PAD_TOKEN])
    test_sequences, test_sequences_lengths = pad_sequences(test_sequences, max_sequence_len, atom_to_one_hot[PAD_TOKEN])

    train_sequences, train_sequences_lengths = zip(*reversed(sorted(zip(train_sequences,
                                                                        train_sequences_lengths),
                                                   key=lambda t: t[1])))
    test_sequences, test_sequences_lengths = zip(*reversed(sorted(zip(train_sequences,
                                                                      train_sequences_lengths),
                                                 key=lambda t: t[1])))

    return (atom_to_one_hot,
            one_hot_to_atom,
            torch.tensor(np.array(train_sequences)).long(),
            torch.tensor(np.array(train_sequences_lengths)).long(),
            torch.tensor(np.array(test_sequences)).long(),
            torch.tensor(np.array(test_sequences_lengths)).long())


def generate_description_sequences_from_dataframe(train_dataframe,
                                                  test_dataframe,
                                                  dictionary_encoder,
                                                  sequence_encoder):
    train_dataframe, test_dataframe = utils.dataframe.remap_columns_with_transform(
        train_dataframe,
        test_dataframe,
        'description',
        'clean_description',
        utils.dataframe.normalize_description
    )

    return generate_description_sequences(train_dataframe['clean_description'],
                                          test_datframe['clean_description'],
                                          dictionary_encoder,
                                          sequence_encoder)
