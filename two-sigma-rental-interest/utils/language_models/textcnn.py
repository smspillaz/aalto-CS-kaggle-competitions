import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self,
                 dictionary_dimension,
                 encoder_dimension,
                 n_filters,
                 filter_sizes,
                 output_dimension,
                 dropout=0.1,
                 pretrained=None):
        super().__init__()

        self.dropout = dropout
        self.embedding = nn.Embedding(dictionary_dimension, encoder_dimension)
        total_embeddings_dimension = sum([e for _, e in categorical_feature_embedding_dimensions])
        self.categorical_feature_embeddings = [
            nn.Embedding(*categorical_feature_embedding_dimensions[f])
            for f in range(len(categorical_feature_embedding_dimensions))
        ]
        for i, embedding in enumerate(self.categorical_feature_embeddings):
            self.add_module('emb_{}'.format(i), embedding)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], encoder_dimension))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], encoder_dimension))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2], encoder_dimension))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dimension)
        self.dropout = nn.Dropout(dropout)

        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, x):
        # x = [batch size, sent len]
        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return F.log_softmax(self.fc(cat), dim=-1)


class TextCNNWithTabularData(nn.Module):
    def __init__(self,
                 dictionary_dimension,
                 encoder_dimension,
                 n_filters,
                 filter_sizes,
                 output_dimension,
                 continuous_features_dimension,
                 categorical_feature_embedding_dimensions,
                 dropout=0.1,
                 pretrained=None):
        super().__init__()

        self.embedding = nn.Embedding(dictionary_dimension, encoder_dimension)
        total_embeddings_dimension = sum([e for _, e in categorical_feature_embedding_dimensions])
        self.categorical_feature_embeddings = [
            nn.Embedding(*categorical_feature_embedding_dimensions[f])
            for f in range(len(categorical_feature_embedding_dimensions))
        ]
        for i, embedding in enumerate(self.categorical_feature_embeddings):
            self.add_module('emb_{}'.format(i), embedding)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], encoder_dimension))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], encoder_dimension))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2], encoder_dimension))
        filtered_dimension = len(filter_sizes) * n_filters
        self.fc = nn.Linear(total_embeddings_dimension +
                            continuous_features_dimension +
                            filtered_dimension,
                            output_dimension)
        self.dropout = nn.Dropout(dropout)

        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, X):
        # x = [batch size, sent len]
        x, continuous_features, categorical_features = X

        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        # Separate out the categorical and continuous features, passing
        # each category through its own embedding
        categorical_embeddings = torch.cat([
            self.categorical_feature_embeddings[f](categorical_features[:,f])
            for f in range(categorical_features.shape[1])
        ], dim=1)
        features = torch.cat((cat, continuous_features, categorical_embeddings), dim=1)

        return F.log_softmax(self.fc(features), dim=-1)

