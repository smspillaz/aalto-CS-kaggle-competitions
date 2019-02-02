import itertools
import pandas as pd

from utils.language_models.descriptions import tokenize_sequences
from utils.language_models.featurize import featurize_sequences_from_sentence_lists

def preview_tokenization(descriptions):
    display(pd.DataFrame(list(zip(*[
        list(descriptions[:10]),
        list(itertools.chain.from_iterable([
            [" ".join(s for s in sequence) for sequence in sequences]
            for sequences in tokenize_sequences(descriptions[:10])
        ]))
    ])), columns=["Original", "Tokenized"]))


def preview_encoded_sentences(sentences):
    auxillary, ((sequences, lengths), ) = featurize_sequences_from_sentence_lists(sentences)
    display(pd.DataFrame(list(zip(*[
        list(sentences),
        [
            ", ".join(str(s) for s in sequence)
            for sequence in sequences.numpy()
        ],
        lengths.numpy()
    ])), columns=["Original", "Encoded", "Length"]))

