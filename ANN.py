from Utils import *


class Layer:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = None
        self.input = None
        self.output = None
        self.bottom = None
        self.params = []
        self.grads = []
        self.cache = None

    def forward(self, input):
        return None

    def backward(self, top_grad):
        return None


class FCLayer(Layer):

    def __init__(self, n_neurons, input_shape):
        Layer.__init__(self, input_shape)
        self.output_shape = (n_neurons,)
        self.n_units = n_neurons
        stddev = np.sqrt(2.0 / (self.input_shape[0] + self.n_units))
        self.weights = np.random.normal(0.0, stddev, size=(self.input_shape[0], self.n_units))
        self.bias = np.ones((self.n_units,)) * 0.01
        self.params = [self.weights, self.bias]
        self.grads = [None, None]

    def forward(self, input):
        self.input = input
        self.output = input.dot(self.weights) + self.bias

    def backward(self, top_grad):
        self.grads[0] = self.input.T.dot(top_grad)
        self.grads[1] = top_grad.sum(axis=0)
        self.bottom = top_grad.dot(self.weights.T)


class ConvLayer(Layer): #Does not pad automatically (performs a valid convolution). Use the pad layer for padding

    def __init__(self, n_filters, filter_shape, stride, dilation, input_shape):
        Layer.__init__(self, input_shape)
        self.filter_shape = filter_shape
        self.stride = stride
        self.dilation = dilation
        self.n_filters = n_filters
        fan_in = input_shape[0] * filter_shape[0] * filter_shape[1]
        fan_out = n_filters * filter_shape[0] * filter_shape[1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        self.filters = np.random.normal(0.0, stddev,
                                        size=(n_filters, self.input_shape[0], filter_shape[0], filter_shape[1]))
        self.bias = np.ones((self.n_filters,)) * 0.01
        self.params = [self.filters, self.bias]
        dilated_shape = ((filter_shape[0] - 1) * dilation + 1, (filter_shape[1] - 1) * dilation + 1)
        self.output_shape = (n_filters,
                             (input_shape[1] - dilated_shape[0]) // stride[0] + 1,
                             (input_shape[2] - dilated_shape[1]) // stride[1] + 1)
        self.grads = [None, None]

    def forward(self, input):
        self.input = input
        self.output = conv2D(input, self.filters, self.dilation, self.stride) + self.bias[np.newaxis, :, np.newaxis,
                                                                                np.newaxis]

    def backward(self, top_grad):
        self.bottom, self.grads[0] = backward_conv2D(top_grad, self.input, self.filters, self.dilation, self.stride)
        self.grads[1] = top_grad.sum(axis=(0, 2, 3))


class MaxPoolLayer(Layer):

    def __init__(self, pool_shape, stride, dilation, input_shape):
        Layer.__init__(self, input_shape)
        self.pool_shape = pool_shape
        self.stride = stride
        self.dilation = dilation
        dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
        self.output_shape = (input_shape[0],
                             (input_shape[1] - dilated_shape[0]) // stride[0] + 1,
                             (input_shape[2] - dilated_shape[1]) // stride[1] + 1)

    def forward(self, input):
        self.input = input
        self.output, self.cache = maxpool2D(input, self.pool_shape, self.dilation, self.stride)

    def backward(self, top_grad):
        self.bottom = backward_maxpool2D(top_grad, self.cache, self.input, self.pool_shape, self.dilation, self.stride)


class ReluLayer(Layer):

    def __init__(self, input_shape):
        Layer.__init__(self, input_shape)
        self.output_shape = self.input_shape

    def forward(self, input):
        self.input = input
        self.output, self.cache = relu(input)

    def backward(self, top_grad):
        self.bottom = backward_relu(top_grad, self.cache)

class FlattenLayer(Layer):

    def __init__(self, input_shape):
        Layer.__init__(self, input_shape)
        self.output_shape = (np.prod(input_shape), )

    def forward(self, input):
        self.input = input
        self.output, self.cache = flatten(input)

    def backward(self, top_grad):
        self.bottom = backward_flatten(top_grad, self.cache)

class PadLayer(Layer):

    def __init__(self, pad_shape, input_shape):
        Layer.__init__(self, input_shape)
        self.output_shape = (self.input_shape[0], self.input_shape[1] + 2 * pad_shape[0], self.input_shape[1] + 2 * pad_shape[1])
        self.pad_shape = pad_shape

    def forward(self, input):
        self.input = input
        self.output = pad2D(input, self.pad_shape)

    def backward(self, top_grad):
        self.bottom = backward_pad2D(top_grad, self.pad_shape)

class SoftmaxCELayer(Layer):

    def __init__(self, input_shape):
        Layer.__init__(self, input_shape)
        self.output_shape = (1, )
        self.truth = None

    def set_truth(self, truth):
        self.truth = truth

    def forward(self, input):
        self.input = input
        self.output, self.cache = softmax_crossentropy(input, self.truth)

    def backward(self, top_grad = 1.0):
        self.bottom = backward_softmax_crossentropy(self.cache, self.truth)


class ANN:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input, truth):
        self.layers[-1].set_truth(truth)
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
        return input

    def run(self, input):
        for layer in self.layers[:-1]:
            layer.forward(input)
            input = layer.output
        return input

    def backward(self):
        top_grad = 1.0
        for layer in self.layers[::-1]:
            layer.backward(top_grad)
            top_grad = layer.bottom

    def trainstep(self, learning_rate=0.01, l2 = 0.0001):
        for layer in self.layers:
            for weight, grad in zip(layer.params, layer.grads):
                weight -= learning_rate * grad + l2 * weight






