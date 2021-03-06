import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import safe_indexing
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks import Callback, Checkpoint, EpochScoring, ProgressBar
from skorch.dataset import Dataset, get_len
from utils.model import sklearn_pipeline_steps
from utils.language_models.descriptions import (
    maybe_cuda,
    maybe_cuda_all
)
from utils.language_models.split import (
    simple_train_test_split_without_shuffle_func
)


class SimpleRNNPredictor(nn.Module):
    """A simple neural net to process text alone."""

    def __init__(self,
                 layers,
                 dictionary_dimension,
                 encoder_dimension,
                 hidden_dimension,
                 output_dimension,
                 dropout=0.05):
        """Initialize and set up layers."""
        super().__init__()

        self.dropout = dropout
        self.n_layers = layers
        self.hidden_dimension = hidden_dimension
        self.word_encoder = nn.Embedding(dictionary_dimension, encoder_dimension)
        self.rnn = nn.LSTM(encoder_dimension,
                           hidden_dimension,
                           num_layers=layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=self.dropout)

        self.decoder = nn.Linear(hidden_dimension * 2 * layers, output_dimension)
        self.class_decoder1 = nn.Linear(hidden_dimension * 2 * layers,
                                        128)
        self.class_decoder2 = nn.Linear(128, 128)
        self.class_decoder3 = nn.Linear(128, output_dimension)

    def forward(self, X):
        """Make a pass through the RNN and combine with input_features."""
        input_sequences, input_sequence_lengths = maybe_cuda_all(*X)

        hidden = maybe_cuda(
            torch.zeros(2 * self.n_layers,
                        input_sequences.shape[0],
                        self.hidden_dimension).float()
        )
        cell_state = maybe_cuda(
            torch.zeros(2 * self.n_layers,
                        input_sequences.shape[0],
                        self.hidden_dimension).float()
        )
        encoded = self.word_encoder(input_sequences)
        padded = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_sequence_lengths, batch_first=True)
        _, (hidden, cell_state) = self.rnn(padded, (hidden, cell_state))

        hidden = hidden.view(hidden.shape[1], hidden.shape[2] * 2 * self.n_layers)
        #decoded = F.dropout(F.sigmoid(self.class_decoder1(hidden)), self.dropout)
        #decoded = F.dropout(F.sigmoid(self.class_decoder2(decoded)), self.dropout)
        #decoded = self.class_decoder3(decoded)
        return F.log_softmax(F.dropout(self.decoder(hidden), self.dropout), dim=-1)


class SimpleRNNTabularDataPredictor(nn.Module):
    """A simple neural net to process text and features."""

    def __init__(self,
                 layers,
                 encoder_dimension,
                 hidden_dimension,
                 dictionary_dimension,
                 continuous_features_dimension,
                 categorical_feature_embedding_dimensions,
                 output_dimension,
                 dropout=0.05):
        """Initialize and set up layers."""
        super().__init__()

        self.dropout = dropout
        self.n_layers = layers
        self.hidden_dimension = hidden_dimension
        self.word_encoder = nn.Embedding(dictionary_dimension, encoder_dimension)
        self.categorical_feature_embeddings = [
            nn.Embedding(*categorical_feature_embedding_dimensions[f])
            for f in range(len(categorical_feature_embedding_dimensions))
        ]
        for i, embedding in enumerate(self.categorical_feature_embeddings):
            self.add_module('emb_{}'.format(i), embedding)
        self.rnn = nn.LSTM(encoder_dimension,
                           hidden_dimension,
                           num_layers=layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=self.dropout)

        total_embeddings_dimension = sum([e for _, e in categorical_feature_embedding_dimensions])
        self.class_decoder1 = nn.Linear(total_embeddings_dimension +
                                        continuous_features_dimension +
                                        hidden_dimension * 2 * layers,
                                        128)
        self.class_decoder2 = nn.Linear(128, 128)
        self.class_decoder3 = nn.Linear(128, output_dimension)

    def forward(self, X):
        """Make a pass through the RNN and combine with input_features."""
        input_sequences, input_sequence_lengths, input_features_continuous, input_features_categorical = maybe_cuda_all(*X)

        # Separate out the categorical and continuous features, passing
        # each category through its own embedding
        categorical_embeddings = torch.cat([
            self.categorical_feature_embeddings[f](input_features_categorical[:,f])
            for f in range(input_features_categorical.shape[1])
        ], dim=1)

        hidden = maybe_cuda(
            torch.zeros(2 * self.n_layers,
                        input_sequences.shape[0],
                        self.hidden_dimension).float()
        )
        cell_state = maybe_cuda(
            torch.zeros(2 * self.n_layers,
                        input_sequences.shape[0],
                        self.hidden_dimension).float()
        )
        encoded = self.word_encoder(input_sequences)
        padded = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_sequence_lengths, batch_first=True)
        _, (hidden, cell_state) = self.rnn(padded, (hidden, cell_state))

        hidden = hidden.view(hidden.shape[1], hidden.shape[2] * 2 * self.n_layers)
        features = torch.cat((hidden, input_features_continuous, categorical_embeddings), dim=1)
        decoded = F.dropout(torch.sigmoid(self.class_decoder1(features)), self.dropout)
        decoded = F.dropout(torch.sigmoid(self.class_decoder2(decoded)), self.dropout)
        decoded = self.class_decoder3(decoded)
        return F.log_softmax(decoded, dim=-1)


class NoToTensorInLossClassifier(NeuralNetClassifier):
    """A simple override of NeuralNetClassifier."""

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Don't call to_tensor on y_true.

        I'm not sure why this fixes the loss=nan issue, but it does -
        something weird to do with the way that to_tensor in skorch is
        implemented.
        """
        # print(torch.max(y_pred, dim=1)[1], y_true)
        return self.criterion_(y_pred, y_true)

    def predict_proba(self, X):
        """Exponentiate the predicted probabilities.

        I don't know why the base model doesn't do this, especially
        since the base loss function is NLLLoss.
        """
        return np.exp(super().predict_proba(X))


class LRAnnealing(Callback):
    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1]['valid_loss_best']:
            net.lr /= 4.0


class CheckpointAndKeepBest(Checkpoint):
    def on_train_end(self, net, **kwargs):
        for i, v in enumerate(net.history[:, self.event_name]):
            if v:
                idx = i

        print('Loading parameters from checkpoint {}'.format(idx))
        net.load_params(checkpoint=self)

