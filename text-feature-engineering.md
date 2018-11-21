# Feature Engineering for Text Data

Word Frequency: If we just take the most frequent words, there are a lot of stopwords

Probably better to only encode words past a certain frequency threshold. We can take the one-hot
encoded vectors and pass a certain number of them into a logistic regressor.

## TF-IDF

You give less weight to words that appear very frequently throughout the entire dataset. Term
Frequency-Inverse Document Frequency.

Now, if your documents are quite small then this is quite a useless weighting. One could plot
the distribution of the TF score for your data such that we can see if we have very frequent
but non-indicative words that don't tell documents apart.

## Word2Vec Methods

### Skip-Gram

You decide on the vocab size as you did before and select words based on the number of appearances.

Words that appear a lot will be taken into account. Then for each word in the vocab, assign
a unique index and with those indices, create a unique onehot vector for them.

We have have fully-connected neural net with a hidden layer that is more compressed in the middle. We
want to "learn" a representation that clusters like words together on certain dimensions. The skip-gram
task is about predicting the context, so it clusters words together that "go together" in the
same context.

We can embed vocabulary to vectors (encoding) or embed vectors to vocabulary (decoding). 

### Gemsim / CBOW model
A training method of Word2Vec. We are predicting the center word for a context.

### Averaging of the word vectors
The default baseline to get a "sentence" representation is to just average the word vectors - of
course this does not preserve ordering information but some dimensions will be higher than average.

### GLoVE

We want to predict the probability ratio of $P(k|ice)$ or $P(k|steam)$ for instance. We predict
the probability of some word appearing *near* another word. This is a global statistic over
the entire dataset as opposed to local word embedding.

## Conclusion

In none of these models did we take into account the structure of the sentence (that is,
the ordering of the words). This is not really a problem with the encodings but of
the models. Note that you can use an LSTM to have variable length inputs to generate the embedding.
