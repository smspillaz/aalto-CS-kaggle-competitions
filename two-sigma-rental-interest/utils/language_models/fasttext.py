import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


class FastText(nn.Module):
    def __init__(self,
                 dictionary_dimension,
                 encoder_dimension,
                 output_dimension,
                 dropout=0.1,
                 pretrained=None):
        super().__init__()

        self.dropout = dropout
        self.embedding = nn.Embedding(dictionary_dimension, encoder_dimension)
        self.fc = nn.Linear(encoder_dimension, output_dimension)

        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, x):
        # x = [batch size, x]
        embedded = self.embedding(x)
        # embedded = [batch size, sent len, emb dim]
        pooled = F.dropout(
            F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) ,
            self.dropout
        )
        # pooled = [batch size, embedding_dim]
        return F.log_softmax(self.fc(pooled), dim=-1)


class FastTextWithTabularData(nn.Module):
    def __init__(self,
                 dictionary_dimension,
                 encoder_dimension,
                 output_dimension,
                 continuous_features_dimension,
                 categorical_feature_embedding_dimensions,
                 dropout=0.1,
                 pretrained=None):
        super().__init__()

        self.dropout = dropout
        self.embedding = nn.Embedding(dictionary_dimension, encoder_dimension)
        total_embeddings_dimension = sum([e for _, e in categorical_feature_embedding_dimensions])
        self.fc = nn.Linear(total_embeddings_dimension +
                            continuous_features_dimension +
                            encoder_dimension,
                            output_dimension)
        self.categorical_feature_embeddings = [
            nn.Embedding(*categorical_feature_embedding_dimensions[f])
            for f in range(len(categorical_feature_embedding_dimensions))
        ]
        for i, embedding in enumerate(self.categorical_feature_embeddings):
            self.add_module('emb_{}'.format(i), embedding)

        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, X):
        x, continuous_features, categorical_features = X
        # x = [batch size, x]
        embedded = self.embedding(x)
        # embedded = [batch size, sent len, emb dim]
        pooled = F.dropout(
            F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) ,
            self.dropout
        )
        # Separate out the categorical and continuous features, passing
        # each category through its own embedding
        categorical_embeddings = torch.cat([
            self.categorical_feature_embeddings[f](categorical_features[:,f])
            for f in range(categorical_features.shape[1])
        ], dim=1)
        features = torch.cat((pooled, continuous_features, categorical_embeddings), dim=1)

        # pooled = [batch size, embedding_dim]
        return F.log_softmax(self.fc(features), dim=-1)
