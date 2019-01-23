import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from fastai.text import Tokenizer, SpacyTokenizer
from torchtext import data

from utils.dataframe import (
    normalize_description,
    remap_columns_with_transform
)

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


def tokens_to_one_hot_lookups(tokens):
    return set_to_one_hot_lookups(sorted(set(tokens) | set([PAD_TOKEN, EOS_TOKEN])))

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

def descriptions_to_token_sequences(tokens, token_to_one_hot):
    return [
        [token_to_one_hot[t] for t in s] + [token_to_one_hot[EOS_TOKEN]]
        for s in tokens
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
    ], np.array([len(s) for s in sequences])


def tokenize_sequences(*sequences):
    tokenizer = Tokenizer(tok_func=SpacyTokenizer)
    return [tokenizer.process_all(s) for s in sequences]


def token_dictionary_seq_encoder(*sequences):
    tokenized_sequences = tokenize_sequences(*sequences)

    tokens_to_one_hot, one_hot_to_tokens = tokens_to_one_hot_lookups(
        list(itertools.chain.from_iterable(itertools.chain.from_iterable(tokenized_sequences)))
    )

    return ((
        tokens_to_one_hot,
        one_hot_to_tokens
    ), tuple(
        descriptions_to_token_sequences(s, tokens_to_one_hot)
        for s in tokenized_sequences
    ))


def reverse_sort_by_length(sequences, lengths):
    """Given two lists of sequences and corresponding lengths, sort descending."""
    return list(zip(*reversed(sorted(list(zip(sequences, lengths)),
                                     key=lambda t: t[1]))))


def generate_description_sequences(dictionary_seq_encoder, *sequences):
    """Given some_descriptions, encode as padded tensors."""
    (atom_to_one_hot, one_hot_to_atom), encoded = dictionary_seq_encoder(*sequences)

    max_sequence_len = max([longest_sequence(s) for s in encoded])
    padded_sequences_and_lengths = [
        reverse_sort_by_length(*pad_sequences(sequences, max_sequence_len, atom_to_one_hot[PAD_TOKEN]))
        for sequences in encoded
    ]

    return (
        (atom_to_one_hot,
         one_hot_to_atom),
        (
            (torch.tensor(s).long(), torch.tensor(l).long())
            for s, l in padded_sequences_and_lengths
        )
    )


def generate_description_sequences_from_dataframe(dictionary_encoder,
                                                  sequence_encoder,
                                                  *dataframes):
    train_dataframe, test_dataframe = remap_columns_with_transform(
        'description',
        'clean_description',
        normalize_description
    )

    return generate_description_sequences(dictionary_encoder,
                                          sequence_encoder,
                                          [df['clean_description'] for df in dataframes])


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def postprocess_sequences(*sequences, postprocessing=None):
    if not postprocessing:
        return sequences

    return tuple(
        [postprocessing(sequence) for sequence in collection]
        for collection in sequences
    )


def insert_average_vectors(vocab):
    zeros = np.zeros(vocab.vectors.shape[1])
    vectors_np = vocab.vectors.numpy()
    zeros_idx = np.array([
        np.all(row == zeros)
        for row in vectors_np
    ])
    average = vectors_np[~zeros_idx].mean(axis=0)
    vectors_np[zeros_idx] = average
    vocab.vectors = torch.tensor(vectors_np)


def torchtext_create_text_vocab(*sequences, vectors=None):
    text = data.Field(tokenize='spacy')

    text.build_vocab(sequences)
    text.vocab.load_vectors(vectors)

    insert_average_vectors(text.vocab)

    return text


def torchtext_process_texts(*sequences, text=None):
    assert text is not None
    return tuple(
        text.process(seq).transpose(0, 1)
        for seq in sequences
    )

