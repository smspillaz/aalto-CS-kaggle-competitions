import numpy as np

from fastai.text import *
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from utils.dataframe import random_oversample_dataframe
from utils.report import generate_classification_report_from_preds


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


def train_ulmfit(learner):
    train_ulmfit_classifier_with_gradual_unfreezing(learner, 1e-2)
    learner.fit_one_cycle(10,
                          slice((1e-2 / 10) / (2.6 ** 4),
                                1e-2 / 10),
                          moms=(0.8, 0.7))

def yield_preds_and_probabilities_from_model(model, dataloader):
    for xb, yb in dataloader:
        probabilities = torch.softmax(model(xb)[0], dim=-1).detach().cpu().numpy()
        preds = np.argmax(probabilities, axis=1)
        yield preds, probabilities


def ulmfit_predict_entire_set(learner, dataloader):
    predictions, probabilities = zip(*list(yield_preds_and_probabilities_from_model(learner.model,
                                                                                    dataloader)))

    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(predictions)
    )


def train_ulmfit_model_and_get_validation_and_test_set_predictions(
    train_dataframe,
    validation_dataframe,
    test_dataframe
):
    with open('vocab.pkl', 'rb') as vocab_f:
        vocab = pickle.load(vocab_f)

    learner = load_ulmfit_classifier_with_transfer_learning_from_data_frame(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
        vocab,
        'renthop_ulmfit',
        'fine_tuned_renthop'
    )
    train_ulmfit(learner)

    validation_probabilities, validation_preds = ulmfit_predict_entire_set(
        learner,
        learner.data.valid_dl
    )
    validation_data, validation_labels = tuple(zip(*[
        x for x in iter(learner.data.valid_dl)
    ]))
    validation_labels = torch.cat(validation_labels).cpu().numpy()
    reconstructed_validation_data = pd.DataFrame(list(itertools.chain.from_iterable([
        [learner.data.valid_ds.reconstruct(d) for d in validation_batch]
        for validation_batch in validation_data
    ])), columns=['description'])
    report = generate_classification_report_from_preds(validation_preds,
                                                       validation_probabilities,
                                                       reconstructed_validation_data,
                                                       validation_labels,
                                                       [0, 1, 2],
                                                       columns=['description'])

    test_probabilities, test_preds = ulmfit_predict_entire_set(
        learner,
        learner.data.test_dl
    )

    return validation_probabilities, test_probabilities
