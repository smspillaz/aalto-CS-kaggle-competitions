from utils.language_models.descriptions import (
    generate_description_sequences,
    token_dictionary_seq_encoder
)

def featurize_sequences_from_sentence_lists(*sentence_lists):
    return generate_description_sequences(
         token_dictionary_seq_encoder,
         *sentence_lists
    )

def featurize_sequences_from_dataframe(*dataframes):
    return featurize_sequences_from_sentence_lists(
         *list(dataframe["description"] for dataframe in dataframes)
    )
