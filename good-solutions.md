# Dmitry - Pump it Up Data Mining the Water Table

Need to source some external data in order to fill in the missing values from
the GPS.

Be aware of the place in which your data is collected, this affects
seasons or significant dates when working with time series data.

Highly correlation features - need lots of regularization to be done.

Logistic regression might not work so well when you have feature interaction.

# Sebastian - Quora

Count some features like number of words, number of unique words, etc.

Instead of replacing missing words with the mean, replace with a random vector with the
same mean and standard deviation as your word embedding model.

Bag of Embeddings: Embedding layer with the glove vectors. EmbeddingBag in PyTorch - it will sum
them for you. Freeze the emebeddings so that the embeddings can't be trained (justification - less
training time, because then you don't have to use the GPU).

F1 Score Threshold: If you just have a 0-1 prediction, it makes sense to try lots of different
thresholds to determine at what probability you decide to classify something as "bad". For instance.
spam/not spam your model might predict 0.2 but it turns out that everything above 0.2 is spam
on the validation set, so you drop your threshold from 0.5 to 0.2.

