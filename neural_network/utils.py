import numpy as np
from warnings import warn


def _im_to_rows(x, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Converts the 4D image to a form such that convolution can be performed via matrix multiplication
    :param x: The image of the dimensions (batch, channels, height, width)
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The transformed image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros((x.shape[0], num_rows * num_cols, filter_shape[1], filter_shape[2], filter_shape[3]), dtype=x.dtype)
    for i in range(num_rows):
        for j in range(num_cols):
            res[:, i * num_cols + j, :, :, :] = x[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation,
                                                j * stride[1]:j * stride[1] + dilated_cols:dilation]
    return res.reshape((res.shape[0], res.shape[1], -1))


def _backward_im_to_rows(top_grad, inp_shape, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Gradient transformation for the im2rows operation
    :param top_grad: The grad from the next layer
    :param inp_shape: The shape of the input image
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The reformed gradient of the shape of the image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros(inp_shape, dtype=top_grad.dtype)
    top_grad = top_grad.reshape(
        (top_grad.shape[0], top_grad.shape[1], filter_shape[1], filter_shape[2], filter_shape[3]))
    for it in range(num_rows * num_cols):
        i = it // num_rows
        j = it % num_rows
        res[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation,
            j * stride[1]:j * stride[1] + dilated_cols:dilation] += top_grad[:, it, :, :, :]
    return res


try:
    from numba.decorators import jit

    _im_to_rows = jit(_im_to_rows)
    _backward_im_to_rows = jit(_backward_im_to_rows)
except ModuleNotFoundError:
    warn("Numba not found, convolutions will be slow.")


def _filter_to_mat(f):
    """
    Converts a filter to matrix form
    :param f: The filter (num_filters, depth, height, width)
    :return: The matrix form of the filter which can be multiplied
    """
    return f.reshape(f.shape[0], -1).T


def _convolved_to_im(im, res_shape):
    """
    Reshapes the convolved matrix to the shape of the image
    :param im: The convolved matrix
    :param res_shape: The expected shape of the result
    :return: The reshaped image
    """
    im = im.transpose((0, 2, 1))
    return im.reshape(im.shape[0], im.shape[1], res_shape[0], res_shape[1])


def conv2d(image, filters, dilation, stride):
    """
    Performs a 2D convolution on the image given the filters
    :param image: The input image (batch, channel, height, width)
    :param filters: The filters (num_filters, depth, height, width)
    :param dilation: The dilation factor for the filter
    :param stride: The stride for convolution
    :return: The convolved image
    """
    filter_shape = filters.shape
    im_shape = image.shape
    dilated_shape = ((filter_shape[2] - 1) * dilation + 1, (filter_shape[3] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, filters.shape, dilation, stride, dilated_shape, res_shape)
    filtmat = _filter_to_mat(filters)
    res = imrow.dot(filtmat)
    return _convolved_to_im(res, res_shape)


def backward_conv2d(top_grad, image, filters, dilation, stride):
    """
    Given the grads from the next op, performs the backward convolution pass
    :param top_grad: The grad from the next op
    :param image: The input image to this operation
    :param filters: The filters for this operation
    :param dilation: The dilation factor for the filter
    :param stride: The stride for the convolution
    :return: A tuple representing the grads wrt the input image and the filters
    """
    filter_shape = filters.shape
    im_shape = image.shape
    dilated_shape = ((filter_shape[2] - 1) * dilation + 1, (filter_shape[3] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, filters.shape, dilation, stride, dilated_shape, res_shape)
    filtmat = _filter_to_mat(filters)
    gradmat = top_grad.reshape((top_grad.shape[0], top_grad.shape[1], -1)).transpose((0, 2, 1))
    filt_grad = np.matmul(imrow.transpose((0, 2, 1)), gradmat).sum(axis=0).T.reshape(filter_shape)
    inp_grad_mat = gradmat.dot(filtmat.T)
    inp_grad = _backward_im_to_rows(inp_grad_mat, image.shape, filters.shape, dilation,
                                    stride, dilated_shape, res_shape)
    return inp_grad, filt_grad


def transposed_conv2d(image, filters, dilation, stride):
    """
    Perform a transposed convolution, which can upscale the image.
    :param image: The input image to upscale
    :param filters: The filters for this operation
    :param dilation: The dilation factor for the filters
    :param stride: The stride for the *forward* convolution
    :return: The return upscaled image
    """
    filter_shape = filters.shape
    im_shape = image.shape
    dilated_shape = ((filter_shape[2] - 1) * dilation + 1, (filter_shape[3] - 1) * dilation + 1)
    res_shape = (im_shape[2] - 1) * stride[0] + dilated_shape[0], (im_shape[3] - 1) * stride[1] + dilated_shape[1]
    image_mat = image.reshape((image.shape[0], image.shape[1], -1)).transpose((0, 2, 1))
    filtmat = _filter_to_mat(filters)
    res_mat = image_mat.dot(filtmat.T)
    return _backward_im_to_rows(res_mat, (image.shape[0], filters.shape[1], *res_shape), filters.shape, dilation,
                                stride, dilated_shape, im_shape[2:])


def backward_transposed_conv2d(top_grad, image, filters, dilation, stride):
    """
    Given the grads from the next operation, performs the backward transposed convolution pass
    :param top_grad: The gradients with respect to the outputs of this operation
    :param image: The input to this operation
    :param filters: The filters used in this operation
    :param dilation: The dilation factor for the filters
    :param stride: The strides for the convolution
    :return: A tuple representing the grads wrt the input image and the filters
    """
    filter_shape = filters.shape
    im_shape = image.shape
    filtmat = _filter_to_mat(filters)
    dilated_shape = ((filter_shape[2] - 1) * dilation + 1, (filter_shape[3] - 1) * dilation + 1)
    gradmat = _im_to_rows(top_grad, filter_shape, dilation, stride, dilated_shape, im_shape[2:])
    image_mat = image.reshape((image.shape[0], image.shape[1], -1)).transpose((0, 2, 1))
    filt_grad = np.matmul(gradmat.transpose((0, 2, 1)), image_mat).sum(axis=0).T.reshape(filter_shape)
    image_grad = gradmat.dot(filtmat)
    return image_grad.transpose((0, 2, 1)).reshape(image.shape), filt_grad


def maxpool2d(image, pool_shape, dilation=1, stride=None):
    """
    Performs the max-pooling operation on the image
    :param image: The image to be maxpooled
    :param pool_shape: The shape of the pool filter
    :param dilation: The dilation of the filter
    :param stride: The stride for the filter (defaults to the shape of the pool
    :return: The pooled image and the argmax cache used for backprop as a tuple
    """
    if stride is None:
        stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape, res_shape)
    imrow = imrow.reshape((imrow.shape[0], imrow.shape[1], im_shape[1], -1))
    maxpooled = np.max(imrow, axis=3).transpose((0, 2, 1))
    maxpooled = maxpooled.reshape((maxpooled.shape[0], maxpooled.shape[1], res_shape[0], res_shape[1]))
    max_indices = np.argmax(imrow, axis=3)
    return maxpooled, max_indices


def backward_maxpool2d(top_grad, max_indices, image, pool_shape, dilation=1, stride=None):
    """
    Performs the backward pass on the max-pool operation
    :param top_grad: The grad from the next op
    :param max_indices: The cache generated in the forward pass
    :param image: The original input image to this op
    :param pool_shape: The shape of the max-pool
    :param dilation: The dilation factor
    :param stride: The stride for the pool (defaults to the shape of the pool)
    :return: The gradient wrt the input image
    """
    if stride is None:
        stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    gradrow = np.zeros((im_shape[0], res_shape[0] * res_shape[1], im_shape[1], pool_shape[0] * pool_shape[1]),
                       dtype=top_grad.dtype)
    gradmat = top_grad.reshape((top_grad.shape[0], top_grad.shape[1], -1)).transpose((0, 2, 1))
    i1, i2, i3 = np.ogrid[:image.shape[0], :res_shape[0] * res_shape[1], :im_shape[1]]
    gradrow[i1, i2, i3, max_indices] = gradmat
    inp_grad = _backward_im_to_rows(gradrow, image.shape, (1, im_shape[1]) + pool_shape, dilation, stride,
                                    dilated_shape, res_shape)
    return inp_grad


def meanpool2d(image, pool_shape, dilation=1, stride=None):
    """
    Performs the mean-pooling operation on the image
    :param image: The image to be mean pooled
    :param pool_shape: The shape of the pool filter
    :param dilation: The dilation of the filter
    :param stride: The stride for the filter (defaults to the shape of the pool
    :return: The pooled image and an empty cache to make it consistent with the max-pool API
    """
    if stride is None:
        stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape, res_shape)
    imrow = imrow.reshape((imrow.shape[0], imrow.shape[1], im_shape[1], -1))
    meanpooled = np.mean(imrow, axis=3).transpose((0, 2, 1))
    meanpooled = meanpooled.reshape((meanpooled.shape[0], meanpooled.shape[1], res_shape[0], res_shape[1]))
    return meanpooled, None


def backward_meanpool2d(top_grad, cache, image, pool_shape, dilation=1, stride=None):
    """
    Performs the backward pass on the mean-pool operation
    :param top_grad: The grad from the next op
    :param cache: Not used
    :param image: The original input image to this op
    :param pool_shape: The shape of the mean-pool
    :param dilation: The dilation factor
    :param stride: The stride for the pool (defaults to the shape of the pool)
    :return: The gradient wrt the input image
    """
    if stride is None:
        stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    gradrow = np.zeros((im_shape[0], res_shape[0] * res_shape[1], im_shape[1], pool_shape[0] * pool_shape[1]),
                       dtype=top_grad.dtype)
    gradmat = top_grad.reshape((top_grad.shape[0],
                                top_grad.shape[1], -1)).transpose((0, 2, 1)) / (pool_shape[0] * pool_shape[1])
    gradrow[:, :, :, :] = gradmat[:, :, :, np.newaxis]
    inp_grad = _backward_im_to_rows(gradrow, image.shape, (1, im_shape[1]) + pool_shape, dilation, stride,
                                    dilated_shape, res_shape)
    return inp_grad


def affine_transform(input_, weight, bias):
    """
    Apply an affine transformation to the input
    :param input_: The input
    :param weight: The weight to be used
    :param bias: The bias to be used
    :return: The transformed input
    """
    return input_.dot(weight) + bias


def backward_affine_transform(top_grad, input_, weight):
    """
    Perform a backward pass on the affine transformation
    :param top_grad: The gradient from the next op
    :param input_: The input used in the forward pass
    :param weight: The weight used in the forward pass
    :return: The gradients for the input, the weight and the bias
    """
    bias_grad = top_grad.sum(axis=0)
    weight_grad = input_.T.dot(top_grad)
    input_grad = top_grad.dot(weight.T)
    return input_grad, weight_grad, bias_grad


def pad2D(image, pad_shape):
    """
    Pads an image with 2D padding of zeros
    :param image: The image to be padded (batch, channel, height, width)
    :param pad_shape: The shape of the symmetric pad (height_pad, width_pad)
    :return: The padded tensor
    """
    return np.pad(image, ((0, 0), (0, 0), (pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1])), mode='constant')


def backward_pad2D(top_grad, pad_shape):
    """
    Performs the backward pass on the pad operation
    :param top_grad: Gradient from the next operation
    :param pad_shape: The pad shape for this op
    :return: The transformed gradient
    """
    return top_grad[:, :, pad_shape[0]:-pad_shape[0], pad_shape[1]:-pad_shape[1]]


def relu(x):
    """
    Performs the ReLU operation on the input tensor
    :param x: The input tensor
    :return: The ReLU'd tensor and a cache used for backprop
    """
    cache = x > 0
    return x * cache, cache


def backward_relu(top_grad, cache):
    """
    Performs the backward pass on the relu operator
    :param top_grad: The gradient from the next operator
    :param cache: The cache from the forward pass
    :return: The gradient wrt the input
    """
    return top_grad * cache


def sigmoid(x):
    """
    Performs the element-wise sigmoid function
    :param x: The input tensor
    :return: The sigmoided tensor
    """
    return 1.0 / (1 + np.exp(-x))


def backward_sigmoid(top_grad, inp_sigmoid):
    """
    Performs the backward pass on the sigmoid operation
    :param top_grad: The grad from the next operation
    :param inp_sigmoid: The output of the forward pass
    :return: The gradient wrt the input of this op
    """
    return top_grad * inp_sigmoid * (1 - inp_sigmoid)


def swish(x):
    """
    Performs the element-wise swish operation
    :param x: The input tensor
    :return: The swished tensor and the sigmoid values
    """
    sigmoid_ = sigmoid(x)
    return x * sigmoid_, sigmoid_


def backward_swish(top_grad, output, sigmoid_):
    """
    Performs the backward pass on the swish operation
    :param top_grad: The gradient from the next operation
    :param output: The output of this operation
    :param sigmoid_: The cache from the forward pass
    :return: The gradient wrt the inputs of this operation
    """
    return top_grad * (sigmoid_ + output * (1 - sigmoid_))


def softmax(x):
    """
    Performs the softmax operation on a 2D tensor
    :param x: The 2D tensor (batch, features)
    :return: The softmaxed tensor
    """
    temp = np.exp(x - x.max(axis=1, keepdims=True))
    res = temp / temp.sum(axis=1, keepdims=True)
    return res


def backward_softmax(top_grad, inp_softmax):
    """
    Performs the backward pass on the softmax operation
    :param top_grad: The gradient from the next operation
    :param inp_softmax: The output of this op
    :return: The gradient wrt the input
    """
    left = inp_softmax[:, :, np.newaxis]
    right = inp_softmax[:, np.newaxis, :]
    sub = left * np.eye(inp_softmax.shape[1])
    mul = np.matmul(left, right)
    res = np.matmul((sub - mul), top_grad[:, :, np.newaxis]).squeeze()
    return res


def crossentropy(x, y):
    """
    Generates the cross-entropy cost
    :param x: The input variable (batch, features)
    :param y: The ground truth (batch, ) (not one-hot)
    :return: The cost
    """
    return np.mean(-np.log(x[np.arange(x.shape[0]), y]))


def backward_crossentropy(top_grad, x, y):
    """
    Performs the backward pass through the crossentropy function
    :param top_grad: The gradient from the next layer
    :param x: The input to this op
    :param y: The ground truth
    :return: The gradient wrt the input
    """
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad


def softmax_crossentropy(x, y):
    """
    Calculates the softmax cross-entropy cost
    :param x: The input variable
    :param y: The ground truth (not one-hot)
    :return: The cost and a the softmaxed values
    """
    s = softmax(x)
    return crossentropy(s, y), s


def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    """
    Backward pass through the softmax crossentropy op
    :param top_grad: The gradient from the next layer.
    :param inp_softmax: The softmax generated in the forward pass
    :param y: The ground truth (not one-hot)
    :return: THe gradient wrt the input
    """
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]


def flatten(x):
    """
    Flattens the tensor into a 2D one
    :param x: The tensor to be flattened
    :return: The flattened tensor and the original shape
    """
    return x.reshape((x.shape[0], -1)), x.shape


def backward_flatten(top_grad, original_shape):
    """
    Performs a backward pass on the flatten operation
    :param top_grad: The gradient from the next op
    :param original_shape: The shape generated during the forward pass
    :return: The gradient wrt the inputs to this op
    """
    return top_grad.reshape(original_shape)


def reshape(x, new_shape):
    """
    Reshape the input to the new shape (preserves the batch)
    :param x: The input
    :param new_shape: The new shape
    :return: The reshaped tensor
    """
    old_shape = x.shape[1:]
    return x.reshape((x.shape[0], *new_shape)), old_shape


def backward_reshape(top_grad, old_shape):
    """
    Perform the backward pass on the reshape operation
    :param top_grad: The gradient from the next layer
    :param old_shape: The old shape
    :return: The gradient for the input
    """
    return top_grad.reshape((top_grad.shape[0], *old_shape))


def tensor_sum(input_):
    """
    Sum the values of the tensor
    :param input_: The tensor
    :return: The sum
    """
    return input_.sum()


def backward_tensor_sum(input_):
    """
    The backward pass for the sum
    :param input_: The input used in the forward pass
    :return: The gradient for the input
    """
    return np.ones_like(input_)


def sse(x, y):
    """
    Sum of squared error between two tensors. Average across the batch.
    :param x: The input tensor
    :param y: The target tensor
    :return: The squared error
    """
    return ((x - y) ** 2).sum() / x.shape[0]


def backward_sse(top_grad, x, y):
    """
    Get the gradient with respect to x.
    :param top_grad: The gradient from the next layer
    :param x: The input
    :param y: The target
    :return: The grad wrt x
    """
    return top_grad * 2 * (x - y) / x.shape[0]
