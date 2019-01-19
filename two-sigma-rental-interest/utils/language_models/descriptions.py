import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from fastai.text import Tokenizer, SpacyTokenizer
from torchtext import data


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
    ], [len(s) for s in sequences]


def token_dictionary_seq_encoder(train_descriptions, test_descriptions):
    tokenizer = Tokenizer(tok_func=SpacyTokenizer)

    tokenized_train_descriptions = tokenizer.process_all(train_descriptions)
    tokenized_test_descriptions = tokenizer.process_all(test_descriptions)

    tokens_to_one_hot, one_hot_to_tokens = tokens_to_one_hot_lookups(
        list(itertools.chain.from_iterable(tokenized_train_descriptions + tokenized_test_descriptions))
    )
    return (
        tokens_to_one_hot,
        one_hot_to_tokens,
        descriptions_to_token_sequences(tokenized_train_descriptions, tokens_to_one_hot),
        descriptions_to_token_sequences(tokenized_test_descriptions, tokens_to_one_hot)
    )


def generate_description_sequences(train_description,
                                   test_description,
                                   dictionary_seq_encoder):
    """Given some train_descriptions and test_descriptions, encode as padded tensors."""
    atom_to_one_hot, one_hot_to_atom, train_sequences, test_sequences = dictionary_seq_encoder(
        train_description,
        test_description,
    )

    max_sequence_len = max([longest_sequence(train_sequences),
                            longest_sequence(test_sequences)])
    train_sequences, train_sequences_lengths = pad_sequences(train_sequences, max_sequence_len, atom_to_one_hot[PAD_TOKEN])
    test_sequences, test_sequences_lengths = pad_sequences(test_sequences, max_sequence_len, atom_to_one_hot[PAD_TOKEN])

    train_sequences, train_sequences_lengths = zip(*reversed(sorted(zip(train_sequences,
                                                                        train_sequences_lengths),
                                                   key=lambda t: t[1])))
    test_sequences, test_sequences_lengths = zip(*reversed(sorted(zip(test_sequences,
                                                                      test_sequences_lengths),
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


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def torchtext_create_text_vocab(texts, vectors=None, preprocessing=None):
    text = data.Field(tokenize='spacy', preprocessing=preprocessing)

    text.build_vocab(texts, vectors=vectors)
    return text


def torchtext_process_df_texts(*dataframes, text=None, field=None):
    assert field is not None
    assert text is not None
    return tuple(
        text.process(list(df[field])).transpose(0, 1)
        for df in dataframes
    )

