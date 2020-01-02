# Numpy Atrous/Dilated and Transposed Convolutions

A pure numpy-based implementation of transposed convolutions which are used for upscaling the tensors and dilated convolutions proposed in [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by Yu et al. The convolutions are performed by matrix multiplications by transforming the image and the filers into matrices. The loops in the transformation routines are numba-compatible and therefore can be compiled using the numba JIT.

## Requirements
- Python 3
- Numpy
- Numba (recommended)
- tqdm

## Creating a neural network

The `Network` class in `network.py` represents a single neural network. The various available layers are also present in `network.py`. These include convolutional layers (with dilation), transposed convolutional layers, pooling layers, fully connected layers, various activations, cross-entropy and squared error losses and various padding/reshaping layers. These layers can be added to the network using the `network.add_layer` function, with the last layer being a loss function. To train a model, you must run the `model.forward` function, which gives you the loss, the `model.backward` function which calculates the derivatives, and the `model.adam_trainstep` function which updates the parameters using the [ADAM](https://arxiv.org/abs/1412.6980) optimizer. For inference, `model.run` can be used.

## Running the tests

The tests for the convolution gradient implementation can by run by

    python conv_tests.py

## Running the networks on MNIST

    python mnist_example.py

The `mnist_example.py` runs two networks on the MNIST dataset present in the repository. The first is a CNN classifier, which should achieve a test accuracy of about 0.97, and the second is a convolution-transposed convolution autoencoder over MNIST trained using a sum of squared errors. The convolution shape arithmetic should be kept in mind when designing such autoencoders.

## Authors
- [Tanmaya Shekhar Dabral](https://github.com/many-facedgod)
