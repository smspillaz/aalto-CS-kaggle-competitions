"""/feedback2vec.py

Given some raw string of feedback and a label (good/bad), build
a model capable of predicting whether the feedback was good or bad.

To do this we have a character encoder which encodes the
characters in the dataset as one-hot encoded letters. We then pass each
character in the stream through an embedding layer, then through a forward
and backward LSTM. The output is then passed to a fully connected
layer which predicts if the feedback was good or bad.

The theory is that we learn representations in the embedding layer which
put the feedback into an appropriate vector space.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import json

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.utils import shuffle


def maybe_cuda(tensor):
    """CUDAifies a tensor if possible."""
    if torch.cuda.is_available():
        return tensor.cuda()

    return tensor.cpu()


class Doc2Vec(nn.Module):
    """Doc2Vec model, based on Tweet2Vec."""

    def __init__(self,
                 embedding_size,
                 hidden_layer_size,
                 vocab_size,
                 output_size,
                 batch_size):
        super().__init__()

        self.hidden_dim = hidden_layer_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # One hidden layers for each direction
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden = (maybe_cuda(torch.randn(2, batch_size, self.hidden_dim)),
                       maybe_cuda(torch.randn(2, batch_size, self.hidden_dim)))
        self.lstm = nn.LSTM(embedding_size,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, output_size)

    def sentence_embedding(self, sentence):
        self.hidden = (maybe_cuda(torch.randn(2, self.batch_size, self.hidden_dim)),
                       maybe_cuda(torch.randn(2, self.batch_size, self.hidden_dim)))
        embeddings = self.embedding(sentence)
        out, self.hidden = self.lstm(embeddings.view(-1, self.batch_size, self.embedding_size),
                                     self.hidden)
        added = self.hidden[0] + self.hidden[1]
        return added / torch.norm(added)

    def forward(self, sentence):
        embedding = self.sentence_embedding(sentence)
        lin = self.linear(embedding.view(-1, self.hidden_dim * 2))
        return F.softmax(lin, dim=1)


def train_model(model, optimizer, epochs, sentence_tensors, label_tensors):
    for epoch in range(epochs):
        total_loss = 0
        loss_criterion = nn.CrossEntropyLoss()

        shuffled_sentence_tensors, shuffled_label_tensors = shuffle(
            sentence_tensors, label_tensors
        )

        for sentence_tensor, label_tensor in tqdm(
            zip(shuffled_sentence_tensors, shuffled_label_tensors),
            total=len(shuffled_label_tensors),
            desc="Processing sentence vectors"
        ):
            optimizer.zero_grad()

            preds = model(sentence_tensor)
            loss = loss_criterion(preds,
                                  label_tensor)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Epoch', epoch, 'loss', total_loss)

        if epoch % 10 == 0:
            with torch.no_grad():
                print("Model Accuracy:",
                      compute_model_accuracy(model,
                                             shuffled_sentence_tensors[0],
                                             shuffled_label_tensors[0]))

    return model


def characters_to_one_hot_lookups(all_characters):
    set_characters = set(all_characters)
    character_to_one_hot = {
        c: i for i, c in enumerate(sorted(set_characters))
    }
    one_hot_to_character = {
        c: i for i, c in enumerate(sorted(set_characters))
    }

    return character_to_one_hot, one_hot_to_character


def character_sequence_to_matrix(sentence, character_to_one_hot):
    return np.array([character_to_one_hot[c] for c in sentence])


def compute_model_accuracy(model, sentence_tensors, label_tensors):
    """A floating point value of how accurate the model was at predicting each label."""
    predictions = np.argmax(model(maybe_cuda(sentence_tensors)).detach().cpu().numpy(), axis=1).flatten()
    labels = label_tensors.detach().cpu().numpy().flatten()
    return len([p for p in (predictions == labels) if p == True]) / len(predictions)


def pad_sentence(sentence, padding):
    truncated = sentence[:padding]
    return truncated + (" " * (padding - len(truncated)))


def to_batches(sentences, batch_size):
    for i in range(len(sentences) // batch_size):
        yield torch.stack([
            sentences[i * batch_size + j]
            for j in range(batch_size)
        ], dim=0)


def documents_to_vectors_model(train_documents,
                               test_documents,
                               labels,
                               epochs,
                               parameters,
                               learning_rate,
                               load=None,
                               save=None):
    """Convert some documents to vectors based on labels."""
    character_to_one_hot, one_hot_to_character = characters_to_one_hot_lookups(
        "".join(train_documents) + "".join(test_documents)
    )
    train_sentence_tensors = list(to_batches([
        maybe_cuda(torch.tensor(character_sequence_to_matrix(pad_sentence(sentence, 500),
                                                             character_to_one_hot), dtype=torch.long))
        for sentence in train_documents
    ], 200))
    
    label_tensors = list(to_batches([maybe_cuda(torch.tensor(i)) for i in labels], 200))

    model = maybe_cuda(Doc2Vec(parameters,
                               parameters * 2,
                               len(character_to_one_hot.keys()), max(labels) + 1,
                               200))

    if not load:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        try:
            train_model(model,
                        optimizer,
                        epochs,
                        train_sentence_tensors,
                        label_tensors)
        except KeyboardInterrupt:
            print("Interrupted, saving current state now")
    else:
        model.load_state_dict(torch.load(load))
    
    if save:
        torch.save(model.state_dict(), save)

    return character_to_one_hot, one_hot_to_character, model


def generate_document_vector_embeddings_from_model(model,
                                                   character_to_one_hot,
                                                   sentences):
    # Generate the embeddings for all of our documents now
    sentence_tensors = list(to_batches([
        maybe_cuda(torch.tensor(character_sequence_to_matrix(pad_sentence(sentence, 500),
                                                             character_to_one_hot), dtype=torch.long))
        for sentence in sentences
    ], 200))
    return np.row_stack([
        model.sentence_embedding(sentence_tensor).view(-1, model.hidden_dim).cpu().numpy()
        for sentence_tensor in sentence_tensors
    ])


def column_to_doc_vectors(train_data_frame,
                          test_data_frame,
                          description_column,
                          target_column,
                          document_vector_column,
                          epochs=100,
                          parameters=40,
                          learning_rate=0.01,
                          load=None,
                          save=None):
    """Convert some description columns to document vector columns."""
    train_descriptions = list(train_data_frame[description_column])
    test_descriptions = list(test_data_frame[description_column])
    labels = list(train_data_frame[target_column])

    character_to_one_hot, one_hot_to_character, model = documents_to_vectors_model(
        train_descriptions,
        test_descriptions,
        labels,
        epochs,
        parameters,
        learning_rate,
        load=load,
        save=save
    )

    train_description_vectors = pd.DataFrame(
        generate_document_vector_embeddings_from_model(
            model,
            character_to_one_hot,
            train_descriptions
        )
    )

    test_description_vectors = pd.DataFrame(
        generate_document_vector_embeddings_from_model(
            model,
            character_to_one_hot,
            test_descriptions
        )
    )

    return (
        pd.concat((train_data_frame, train_description_vectors), axis=1),
        pd.concat((test_data_frame, test_description_vectors), axis=1)
    )

