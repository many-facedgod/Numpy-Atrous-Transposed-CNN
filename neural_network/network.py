import itertools

from .utils import *


class Layer:
    """
    A superclass for all layers
    """
    def __init__(self):
        self.built = False
        self.input_shape = self.output_shape = None
        self.params = []
        self.grads = []

    def build(self, input_shape):
        """Initialize the actual parameters. To be called on the first forward pass"""
        raise NotImplementedError

    def forward(self, *args):
        """The forward pass through the layer. Initializes the params if it's the first call. Returns the output."""
        raise NotImplementedError

    def backward(self, top_grad):
        """The backward pass through the layer to calculate the gradients. Returns the gradient wrt the input."""
        raise NotImplementedError


class FCLayer(Layer):
    """
    A fully connected layer.
    """
    def __init__(self, n_units):
        """
        :param n_units: The number of hidden units
        """
        Layer.__init__(self)
        self.n_units = n_units

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (self.n_units,)
        stddev = np.sqrt(2.0 / (self.input_shape[0] + self.n_units))
        self.weights = np.random.normal(0.0, stddev, size=(self.input_shape[0], self.n_units))
        self.bias = np.ones((self.n_units,)) * 0.01
        self.params = [self.weights, self.bias]
        self.grads = [np.empty_like(param) for param in self.params]
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = affine_transform(input_, self.weights, self.bias)
        return self.output

    def backward(self, top_grad):
        input_grad, weight_grad, bias_grad = backward_affine_transform(top_grad, self.input, self.weights)
        self.grads[0][...] = weight_grad
        self.grads[1][...] = bias_grad
        self.bottom_grad = input_grad
        return self.bottom_grad


class ConvLayer(Layer):
    """A convolutional layer that performs a valid convolution on the input."""

    def __init__(self, n_filters, filter_shape, stride=(1, 1), dilation=1):
        """
        :param n_filters: The number of convolution filters
        :param filter_shape: The shape of each filter
        :param stride: The stride for convolving
        :param dilation: The dilation factor for the filters
        """
        Layer.__init__(self)
        self.filter_shape = filter_shape
        self.stride = stride
        self.dilation = dilation
        self.n_filters = n_filters

    def build(self, input_shape):
        self.input_shape = input_shape
        fan_in = input_shape[0] * self.filter_shape[0] * self.filter_shape[1]
        fan_out = self.n_filters * self.filter_shape[0] * self.filter_shape[1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        self.filters = np.random.normal(0.0, stddev,
                                        size=(self.n_filters, self.input_shape[0],
                                              self.filter_shape[0], self.filter_shape[1]))
        self.bias = np.ones((self.n_filters,)) * 0.01
        self.params = [self.filters, self.bias]
        dilated_shape = ((self.filter_shape[0] - 1) * self.dilation + 1, (self.filter_shape[1] - 1) * self.dilation + 1)
        self.output_shape = (self.n_filters,
                             (input_shape[1] - dilated_shape[0]) // self.stride[0] + 1,
                             (input_shape[2] - dilated_shape[1]) // self.stride[1] + 1)
        self.grads = [np.empty_like(param) for param in self.params]
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = conv2d(input_, self.filters, self.dilation, self.stride) + self.bias[np.newaxis, :, np.newaxis,
                                                                                 np.newaxis]
        return self.output

    def backward(self, top_grad):
        self.bottom_grad, self.grads[0][...] = backward_conv2d(top_grad, self.input, self.filters,
                                                               self.dilation, self.stride)
        self.grads[1][...] = top_grad.sum(axis=(0, 2, 3))
        return self.bottom_grad


class TransposedConvLayer(Layer):
    """
    A layer that performs a transposed convolution. The output shape will be:

        stride * (inp_shape - 1) + dilation * (filter_shape - 1) + 1

    This layer can be used to upscale a tensor.
    """

    def __init__(self, n_filters, filter_shape, stride, dilation=1):
        """
        :param n_filters: The number of convolution filters (channels expected in the output of this layer)
        :param filter_shape: The shape of each filter
        :param stride: The stride for forward convolving
        :param dilation: The dilation factor for the filters
        """
        Layer.__init__(self)
        self.filter_shape = filter_shape
        self.stride = stride
        self.dilation = dilation
        self.n_filters = n_filters

    def build(self, input_shape):
        self.input_shape = input_shape
        fan_in = input_shape[0] * self.filter_shape[0] * self.filter_shape[1]
        fan_out = self.n_filters * self.filter_shape[0] * self.filter_shape[1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        self.filters = np.random.normal(0.0, stddev,
                                        size=(self.input_shape[0], self.n_filters,
                                              self.filter_shape[0], self.filter_shape[1]))
        self.bias = np.ones((self.n_filters,)) * 0.01
        self.params = [self.filters, self.bias]
        dilated_shape = ((self.filter_shape[0] - 1) * self.dilation + 1, (self.filter_shape[1] - 1) * self.dilation + 1)
        self.output_shape = (self.n_filters,
                             (input_shape[1] - 1) * self.stride[0] + dilated_shape[0],
                             (input_shape[2] - 1) * self.stride[1] + dilated_shape[1])
        self.grads = [np.empty_like(param) for param in self.params]
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = transposed_conv2d(input_, self.filters, self.dilation, self.stride) + self.bias[np.newaxis, :,
                                                                                                      np.newaxis,
                                                                                                      np.newaxis]
        return self.output

    def backward(self, top_grad):
        self.bottom_grad, self.grads[0][...] = backward_transposed_conv2d(top_grad, self.input, self.filters,
                                                                          self.dilation, self.stride)
        self.grads[1][...] = top_grad.sum(axis=(0, 2, 3))
        return self.bottom_grad


class Pool2DLayer(Layer):
    """A pooling layer that picks out the maximum element."""

    def __init__(self, pool_shape, stride=None, dilation=1, pool_type='max'):
        """
        :param pool_shape: The shape for pooling.
        :param stride: The stride for the filter. If None, taken to be the same as the pool_shape.
        :param dilation: The dilation factor for the filter.
        """
        Layer.__init__(self)
        self.pool_shape = pool_shape
        self.stride = stride if stride is not None else pool_shape
        self.dilation = dilation
        self.pool_type = pool_type
        self.forward_pool_fn = maxpool2d if pool_type == 'max' else meanpool2d
        self.backward_pool_fn = backward_maxpool2d if pool_type == 'max' else backward_meanpool2d

    def build(self, input_shape):
        self.input_shape = input_shape
        dilated_shape = ((self.pool_shape[0] - 1) * self.dilation + 1, (self.pool_shape[1] - 1) * self.dilation + 1)
        self.output_shape = (input_shape[0],
                             (input_shape[1] - dilated_shape[0]) // self.stride[0] + 1,
                             (input_shape[2] - dilated_shape[1]) // self.stride[1] + 1)
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = self.forward_pool_fn(input_, self.pool_shape, self.dilation, self.stride)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = self.backward_pool_fn(top_grad, self.cache, self.input, self.pool_shape,
                                                 self.dilation, self.stride)
        return self.bottom_grad


class ReluLayer(Layer):
    """An activation layer that activates with the ReLU activation."""

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = relu(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_relu(top_grad, self.cache)
        return self.bottom_grad


class SigmoidLayer(Layer):
    """An activation layer that activates with the sigmoid activation."""

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = sigmoid(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_sigmoid(top_grad, self.output)
        return self.bottom_grad


class FlattenLayer(Layer):
    """A layer that flattens all the dimensions except the batch."""

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = flatten(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_flatten(top_grad, self.cache)
        return self.bottom_grad


class ReshapeLayer(Layer):
    """A layer that reshapes the tensor to a new shape (preserves the batch dimension)."""

    def __init__(self, new_shape):
        """
        :param new_shape: The new shape to reshape to.
        """
        Layer.__init__(self)
        self.new_shape = new_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.new_shape
        assert np.prod(self.new_shape) == np.prod(self.input_shape), (f'Input shape {input_shape} not compatible with '
                                                                      f'the given shape {self.new_shape}')
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = reshape(input_, self.new_shape)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_reshape(top_grad, self.cache)
        return self.bottom_grad


class Pad2DLayer(Layer):
    """Pads a 2D image with zeros."""

    def __init__(self, pad_shape):
        """
        :param pad_shape: A tuple representing the height and the width padding
        """
        Layer.__init__(self)
        self.pad_shape = pad_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (self.input_shape[0], self.input_shape[1] + 2 * self.pad_shape[0],
                             self.input_shape[1] + 2 * self.pad_shape[1])
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = pad2D(input_, self.pad_shape)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_pad2D(top_grad, self.pad_shape)
        return self.bottom_grad


class SoftmaxCELayer(Layer):
    """Calculates the softmax-crossentropy loss of the given input logits wrt some truth value."""

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = ()
        self.built = True

    def forward(self, input_, truth):
        """
        :param input_: The logits
        :param truth: The indices of the correct classification
        :return: The calculated loss
        """
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.truth = truth
        self.output, self.cache = softmax_crossentropy(input_, self.truth)
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = backward_softmax_crossentropy(top_grad, self.cache, self.truth)
        return self.bottom_grad


class SSELayer(Layer):
    """Calculates the sum of squared error between the input and the truth value. """

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = ()
        self.built = True

    def forward(self, input_, truth):
        """
        :param input_: The logits
        :param truth: The indices of the correct classification
        :return: The calculated loss
        """
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.truth = truth
        self.output = sse(input_, self.truth)
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = backward_sse(top_grad, self.input, self.truth)
        return self.bottom_grad


class Network:
    """A sequential neural network"""

    def __init__(self):
        self.layers = []
        self.params = []
        self.grads = []
        self.optimizer_built = False

    def add_layer(self, layer):
        """
        Add a layer to this network. The last layer should be a loss layer.
        :param layer: The Layer object
        :return: self
        """
        self.layers.append(layer)
        return self

    def forward(self, input_, truth):
        """
        Run the entire network, and return the loss.
        :param input_: The input to the network
        :param truth: The ground truth labels to be passed to the last layer
        :return: The calculated loss.
        """
        input_ = self.run(input_)
        return self.layers[-1].forward(input_, truth)

    def run(self, input_, k=-1):
        """
        Run the network for k layers.
        :param k: If positive, run for the first k layers, if negative, ignore the last -k layers. Cannot be 0.
        :param input_: The input to the network
        :return: The output of the second last layer
        """
        k = len(self.layers) if not k else k
        for layer in self.layers[:min(len(self.layers) - 1, k)]:
            input_ = layer.forward(input_)
        return input_

    def backward(self):
        """
        Run the backward pass and accumulate the gradients.
        """
        top_grad = 1.0
        for layer in self.layers[::-1]:
            top_grad = layer.backward(top_grad)

    def adam_trainstep(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, l2=0.):
        """
        Run the update step after calculating the gradients
        :param alpha: The learning rate
        :param beta_1: The exponential average weight for the first moment
        :param beta_2: The exponential average weight for the second moment
        :param epsilon: The smoothing constant
        :param l2: The l2 decay constant
        """
        if not self.optimizer_built:
            self.params.extend(itertools.chain(*[layer.params for layer in self.layers]))
            self.grads.extend(itertools.chain(*[layer.grads for layer in self.layers]))
            self.first_moments = [np.zeros_like(param) for param in self.params]
            self.second_moments = [np.zeros_like(param) for param in self.params]
            self.time_step = 1
            self.optimizer_built = True
        for param, grad, first_moment, second_moment in zip(self.params, self.grads,
                                                            self.first_moments, self.second_moments):
            first_moment *= beta_1
            first_moment += (1 - beta_1) * grad
            second_moment *= beta_2
            second_moment += (1 - beta_2) * (grad ** 2)
            m_hat = first_moment / (1 - beta_1 ** self.time_step)
            v_hat = second_moment / (1 - beta_2 ** self.time_step)
            param -= alpha * m_hat / (np.sqrt(v_hat) + epsilon) + l2 * param
        self.time_step += 1
