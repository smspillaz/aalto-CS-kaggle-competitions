# Deep Learning Fundamentals

Logistic Regression: $P(t|x, w) = \sigma(Wx + b)$

Most of your time the data is not linearly separable. What about polynomials
and weights on the polynomial?

P(t - 1|x) = \sigma(w_0 + w_1x + w_2x^2)$

What we did was transform the data vector into a new feature vector. We created a polynomial
line that fitted the data or rather transformed the data space such that the data was
linearly separable in the new space.

However, curse of dimensionality.

RBFs: Transform the initial feature vectors into something else that is non-linear:

$\phi(1, \phi_1(x), ..., \phi_M(x))$

Now the data is linear to $\phi$ but not to $x$. For instance, the Gaussian
RBF

$$e^{-\frac{1}{2}||x - \mu||^2}$$

Eg, the function has a bell curve and has a radius that you can actually measure.

A good RBF strategy is to use clustering to find where the data lives. The thing is that this
requires us to a priori choose the form of feature vector.

A neural net just tries to learn the representation intself.

## Multilayer Perceptron

We get an input $x$, transform it linearly, transform it by a nonlinear function, then keep
doing that (stacking layers).

This is called forward propagation. The nonlinearities are called activations.

## Objective Function

Depends on what you want to measure. For instance, to compare a probability distribution
use `CrossEntropyLoss` (which is just `NLLLoss(LogSoftmax(x))`)

## Backprop

Basically just use the chain rule to compute the gradient with respect to all the weights.

## Activation Functions

Sigmoid: Gradient goes to zero at around -4 to 4 ($\sigma(x)$ goes from 0 to 1)

Tanh: Goes from -1 to 1

Softmax: Generalization of sigmoid for multiclass ($\frac{e^x}{\sum e^x}$)

ReLU: Can lead to dead ReLUs which leans that your learning becomes very slow, dead neurons
propagate to other layers.

LeakyReLU: Everything becomes close to zero instead of zero (even if you use LeakyReLU it
just ends up being worse).

SeLU: $\log (1 + e^x)$: Doesn't work

## Challenges in Deep Learning

Ill-Condition of the Hessian: A gradient descent step adds to the cost $\frac{1}{2}\epsilon^2 g^THg - \epislong^Tg$ -
if the first term is greater than the first, then learning becomes very slow. When there is strong
curvature you need to shrink the learning rate in order to not fall from a cliff.

Local Minima: Latent variable models can have latent variables exchanged with each other and
nothing changes. This is called weight space symmetry.

Long Term Dependencies: If you have a very deep network and eigendecompose the weights, then
if the eigenvalues > 1 then the gradients will become very large (basically infinite)

## Gradient Descent

Annealing: Decrease the learning rate as time goes on (eg $\frac{{\tau}{\t}$).

Momentum: Apply momentum to the learning rate (if we have large change in gradients, move slowly,
if we have small change in gradients, move quickly).

 - Nesterov Momentum: First make a big jump in the direction of the previous accumulated gradient,
   then measure the gradient where you end up and make a correction. It goes a bit faster and avoids
   problematic areas of the space.
 - Adagrad: Tries different learning rates for every parameter
    - Sample the minibatch, compute the gradient, then accumulate teh squared gradient $r \leftarrow r + g \elem g$
    - Then, each parameter has its own learning rate when applied to the gradient.
 - RMSProp:
 - Adam: Adds momentum to RMSprop.
    - Accumulates the gradient using a convex combination
    - Then does the same for the squared gradient.
    - Updates the biased first moemnt estimate and biased second moment estimate.
    - Then we correct for bias
    - So we don't slide like a ball, but we have some friction.

## Initialization
 - Xavier Initialization (works well for Sigmoid, Tanh), constrains
   the parameters to not go beyond -4, +4 such that the gradient does
   not become zero.
 - He initialization (works well for ReLU), non-negative initialization

## Regularization
The deeper your network, the more you have to regularize it in order to avoid overfitting.

L2 Regularization: You introduce the squared value of the weight into the error function - this causes
weights that are very large to be penalized in relation to the error function.

The effect of weight decay is to rescale the weights if they are too big, so it rescales
the weights along the eigenvectors of the Hessian matrix. This adds variance to the input
features and shrinks the weights on features whose covariance with the output target is compared
to the added variance.

L1 Regularization: Can lead to sparse solutions, this is actually a form of feature selection.

Dropout: Just kill certain activations randomly. Prevents overuse of one activation. Shrinks the
effective capacity of the model, so you need to increase your hidden unit size for more dropout.

Parameter Sharing: Try to make weights equal by forcing this as a penalty. For instance, Siamese Networks
(can be used in Autoencoders).

Data Augmentation: Transforms for images etc.

Multitask Learning: You have different tasks to optimize against.

Batch Normalization: Normalizes every activation for every layer, reduces the amount by which the hidden
units shift around. Each layer has more learning independency and there are no extreme activations and adds some noise to each
activation layer, so acts as a slight regularizer.

# Architectures

RNN: Works for a sequence of inputs. You have shared weights for each timestep. That said, catastrophic forgetting.

CNNs: Feedforward networks don't scale well due to the large input size. Three types of layers:
 - Convolutional: Shared weights that are applied across an array.
 - Pooling
 - Fully Connected 

# Normalization

Batch Norm vs Layer Norm:
 - In BatchNorm you normalize over the feature dimensions
 - In LayerNorm you normalize over the vectors themselves (which is weird, but it works)

In BatchNorm you need to have a batch and does not work well if you change
the batch size all the time.

In LayerNorm you don't need to have a batch (or rather it does not matter).

Learning rate is affected by your batch size (because you're taking a step at different intervals).
