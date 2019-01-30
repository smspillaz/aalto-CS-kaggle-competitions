# Attention and BERT

They are usually good for language modelling - create an output probability distribution,
memory is pretty limited, can't retain information for a long time. Starts off realy good
but once the length of the setnences increase to a point because of the vanishing gradient
issue.

The issue with RNNS for machine translation is that it needs to see the entire
sequence first until it can start decoding.

## Auto-regressive CNNs

Look at the first word, then second word, then fourth word, then 8th word,
than 16th word. Problem: This isn't very logical - local information in
sentences is not purely positional.

## Self-Attention models

You don't care that much about the exact position of the word or token that you are looking
for.

Compare Convolution to Self-Attention - in convolutions we pay similar attention to
all the context words, but with self-attention we can figure out what to pay attention
to.

## Transformer Architecture

We don't need recursion, we don't need convolution, attention is enough.

In RNN models you just the previous history and the input - an attention model
asks "how much attention should I pay to each previous input", but in an RNN you
are recursing back through the *entire* history. Eg, in the decoding phase
you never see hidden layers but only the final hidden layers.

### Attention

Attention is just - how much attention do you pay to each of the tokens? You
have a key-value based retrieval.

$A(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$

So basically given a query vector, we work out how many similar key vectors there are.

Softmaxing them ends up with a probability - a probability distribution over your keys -
which of the actual values in your value matrix do you pay attention to?

Need to scale by $\sqrt{d_k}$ - when you have longer vectors the dot product can become
very big - scale by the dimensionality of your space.

### What are V, Q and K?

Start with x, which is basically the word2vec embedding and position encoding (sum).

Then we pass $x$ through three different linear layers which gives you $Q, V$ and $K$.

The $K$ and $Q$ will have the same dimension and $V$ can have a dimension of $d_v$

### Multi-Head attention

Lots of different understanding of what you pay attention to.

Multiple attention layers in parallel. You have $3h$ different linear
layers which project your $x$ into each of $Q$, $V$ and $K$.

Then concatenate each of them together and classify what to pay attention to
from there. This is now your "hidden" state and it goes to your decoder block.

### Decoder Block

Now, your values and keys are from the things it has already seen. The query comes
from the current word. Works out which part of the encoder to pay attention to.

Masked-Decoder Self-Attention: RNNs can't do bidirectional attention, similar to RNNs
in that we don't look forward. It can't look at the next word since it hasn't generated
it yet!

### Visualizing Attention

For instance "the animal didn't cross the street because it was too tired" (it, pays
attention to "animal"). Compare "the animal didn't cross the street because it was too
wide" ("it" now pays attention to "wide"

### Positional Encodings

Not learned positional encodings, instead we have positional encodings given by
a sine wave to tell the relevant position in a sentence.

# BERT

Bidirectionality is a big thing - if you condition both on the left and right hand side
of the sentence - if you look at the next word, then you can look at the previous word
from the next word and thus see yourself implicitly.

## ELMo
ELMo does a forward and backward pass - use something in a sentence to get the contextual embedding.

## Transformer GPT

Uses the transformer's decoder, train on a regular langauge model task, conditioned only on left
context, can learn much more. Can learn on generic datasets (completely unsupervised).

## Bidirectionality

Instead of predicting the next word - remove some words from a sentence and fill in the
blanks.

## Fine-Turning and BERT

## BERT Input

You input first [CLS] (classification), then you have a sentence separator [SEP]
and then the mask.

### When can you use BERT?

### Pre-training BERT

Once you get the pre-training then the fine-tuning is fast enough.

Masked language model (MLM) - randomly mask some of hte tokens from the input and the objective
is to predict the original vocab id of the masked word based only on its context.

 -> 80% of the time (Replace word with [MASK] toekn)
 -> 10% of the time (Replace word with random word)
 -> 10% of the time (Keep word unchanged)

There is also a "next sentence prediction" task - where we jointly pre-train text pair representations

Input: [CLS] the man went to [MASK] store [SEP] he bought a callon [MASK] milk [SEP]

### Fine-Tuning BERT

Introduce minimal task-specific parameters.

First you adapt your input into the BERT format, then find the best hyperparameters.

!! Try different hyperparameters in a grid search: 
 - Batch Size: 16, 32
 - Learning Rate: 6e-5, 3e-5, 3e-5, 2e-5
 - Number of epochs 3, 4

