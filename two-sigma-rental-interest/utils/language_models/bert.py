import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm_notebook as tqdm
from tqdm import trange

from skorch.classifier import NeuralNetClassifier

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PreTrainedBertModel
from pytorch_pretrained_bert.optimization import BertAdam

from utils.language_models.split import ordered_train_test_split_with_oversampling

from utils.language_models.descriptions import (
    maybe_cuda,
    maybe_cuda_all
)

def bert_featurize_sentences(sentences, max_len, tokenizer):
    for sentence in sentences:
        tokens = ["[CLS]"] + tokenizer.tokenize(sentence)[:max_len - 2] + ["[SEP]"]
        padding = [0] * (max_len - len(tokens))
        segment_ids = [0] * len(tokens) + padding
        input_ids = tokenizer.convert_tokens_to_ids(tokens) + padding
        input_mask = [1] * len(tokens) + padding

        yield (torch.tensor(input_ids, dtype=torch.long),
               torch.tensor(input_mask, dtype=torch.long),
               torch.tensor(segment_ids, dtype=torch.long))

def bert_featurize_data_frame(data_frame, max_len, tokenizer):
    yield from bert_featurize_sentences(data_frame['clean_description'],
                                        max_len,
                                        tokenizer)

def bert_featurize_data_frames(bert_model, *dataframes):
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

    return (
        list(bert_featurize_data_frame(df, 100, tokenizer))
        for df in dataframes
    )


BERT_MODEL = "bert-base-uncased"
TRAIN_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 15
LEARNING_RATE = 5e-5
WARMUP_PROPORTION = 0.1


class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification, sentences only
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self,
                 config,
                 continuous_features_dimension=None,
                 categorical_feature_embedding_dimensions=None,
                 output_dimension=None,
                 num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,
                                    num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, X):
        (input_ids,
         attention_mask,
         token_type_ids) = tuple(maybe_cuda(t) for t in X)

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        features = pooled_output
        features = self.dropout(features)
        logits = self.classifier(features)

        return F.log_softmax(logits, dim=-1)


def create_bert_model(bert_model, num_labels):
    return BertForSequenceClassification.from_pretrained(
        bert_model,
        num_labels=num_labels
    )


class BertForSequenceClassificationWithTabularData(PreTrainedBertModel):
    """BERT model for classification, with tabular data
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self,
                 config,
                 continuous_features_dimension=None,
                 categorical_feature_embedding_dimensions=None,
                 output_dimension=None,
                 num_labels=2):
        super(BertForSequenceClassificationWithTabularData, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        self.categorical_feature_embeddings = [
            nn.Embedding(*categorical_feature_embedding_dimensions[f])
            for f in range(len(categorical_feature_embedding_dimensions))
        ]
        for i, embedding in enumerate(self.categorical_feature_embeddings):
            self.add_module('emb_{}'.format(i), embedding)

        total_embeddings_dimension = sum([
            e for _, e in categorical_feature_embedding_dimensions
        ])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(total_embeddings_dimension +
                                    continuous_features_dimension +
                                    config.hidden_size,
                                    num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, X):
        (input_ids,
         attention_mask,
         token_type_ids,
         input_features_continuous,
         input_features_categorical) = tuple(maybe_cuda(t) for t in X)

        # Separate out the categorical and continuous features, passing
        # each category through its own embedding
        categorical_embeddings = torch.cat([
            self.categorical_feature_embeddings[f](input_features_categorical[:,f])
            for f in range(input_features_categorical.shape[1])
        ], dim=1)

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        features = torch.cat((pooled_output, input_features_continuous, categorical_embeddings), dim=1)
        features = self.dropout(features)
        logits = self.classifier(features)

        return F.log_softmax(logits, dim=-1)


def create_bert_model_with_tabular_features(bert_model,
                                            continuous_features_dimension,
                                            categorical_feature_embedding_dimensions,
                                            num_labels):
    return BertForSequenceClassificationWithTabularData.from_pretrained(
        bert_model,
        continuous_features_dimension=continuous_features_dimension,
        categorical_feature_embedding_dimensions=categorical_feature_embedding_dimensions,
        num_labels=num_labels
    )


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class BertClassifier(NeuralNetClassifier):
    """A simple override of NeuralNetClassifier."""

    def __init__(self, *args, num_train_steps=1, num_labels=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_step = 0
        self._global_training_step = 0
        self._total_training_steps = num_train_steps
        self._num_labels = num_labels
        self._batch_step = 0

    def set_total_training_steps(self, total_training_steps):
        self._total_training_steps = total_training_steps

    def on_epoch_begin(self, *args, **kwargs):
        super().on_epoch_begin(*args, **kwargs)
        self._batch_step = 0

    def on_batch_begin(self, *args, **kwargs):
        super().on_batch_begin(*args, **kwargs)
        self._batch_step += 1

    def train_step(self, Xi, yi, **fit_params):
        step_accumulator = self.get_train_step_accumulator()
        step = self.train_step_single(Xi, yi, **fit_params)
        step_accumulator.store_step(step)

        if (self._batch_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = LEARNING_RATE * warmup_linear(
                self._global_training_step / self._total_training_steps,
                WARMUP_PROPORTION
            )
            for param_group in self.optimizer_.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer_.step()
            self.optimizer_.zero_grad()
            self._global_training_step += 1

        return step_accumulator.get_step()

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Don't call to_tensor on y_true.

        I'm not sure why this fixes the loss=nan issue, but it does -
        something weird to do with the way that to_tensor in skorch is
        implemented.
        """
        # print(torch.max(y_pred, dim=1)[1], y_true)
        return self.criterion_(y_pred.view(-1, self._num_labels),
                               y_true.view(-1)) / GRADIENT_ACCUMULATION_STEPS

    def _get_params_for_optimizer(self, prefix, named_parameters):
        """We have our own logic here for getting params.

        We want to disable weight decay on all the normalization
        layers.
        """
        kwargs = self._get_params_for(prefix)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        args = [optimizer_parameters]
        return args, kwargs

    def predict_proba(self, X):
        """Exponentiate the predicted probabilities.

        I don't know why the base model doesn't do this, especially
        since the base loss function is NLLLoss.
        """
        return np.exp(super().predict_proba(X))

