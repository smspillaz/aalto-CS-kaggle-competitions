from fastai.text import *
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from utils.dataframe import random_oversample_dataframe


def load_ulmfit_classifier_with_transfer_learning_from_data_frame(training_dataframe,
                                                                  validation_dataframe,
                                                                  test_dataframe,
                                                                  vocab,
                                                                  path,
                                                                  encoder_weights_name,
                                                                  dropout_multiplier=0.7):
    """Train ULMFiT classification head given data frames and a vocabulary.

    This automatically oversamples the text data to ensure that the classes
    are balanced.

    This loads the encoder weights from path/encoder_weights_name.
    """
    bs = 48

    data_clas = TextDataBunch.from_df(path=path,
                                      train_df=training_dataframe,
                                      test_df=test_dataframe,
                                      valid_df=validation_dataframe,
                                      text_cols=['clean_description'],
                                      label_cols=['label_interest_level'],
                                      bs=bs,
                                      vocab=vocab)

    learn = text_classifier_learner(data_clas, drop_mult=dropout_multiplier)
    learn.load_encoder(encoder_weights_name)
    learn.freeze()

    return learn


def save_and_load_weights(learn, name):
    learn.save(name)
    learn.load(name)


def train_ulmfit_classifier_with_gradual_unfreezing(learn, lr):
    """Train the ULMFiT classifier with gradual unfreezing.

    The learning rate :lr: will be lowered during training as we unfreeze more
    and more weights.
    """
    learn.fit_one_cycle(2, lr, moms=(0.8,0.7))
    save_and_load_weights(learn, "first")

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(lr / (2.6 ** 4), lr), moms=(0.8, 0.7))
    save_and_load_weights(learn, "second")

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice((lr / 2) / (2.6 ** 4), lr / 2), moms=(0.8, 0.7))
    save_and_load_weights(learn, "third")

    learn.unfreeze()
    learn.fit_one_cycle(2, slice((lr / 10) / (2.6 ** 4), lr / 10), moms=(0.8, 0.7))
    save_and_load_weights(learn, "unfrozen")

    return learn
