import numpy as np
from warnings import warn


def im2rows(x, filter_shape, dilation, stride, dilated_shape, res_shape):
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


def backward_im2rows(top_grad, x, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Gradient transformation for the im2rows operation
    :param top_grad: The grad from the next layer
    :param x: The image of the dimensions (batch, channels, height, width)
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The reformed gradient of the shape of the image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros(x.shape, dtype=x.dtype)
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

    im2rows = jit(im2rows)
    backward_im2rows = jit(backward_im2rows)
except:
    warn("Numba not working, convolution will be slow.")


def filter2mat(f):
    """
    Converts a filter to matrix form
    :param f: The filter (num_filters, depth, height, width)
    :return: The matrix form of the filter which can be multiplied
    """
    return f.reshape(f.shape[0], -1).T


def convolved2im(im, res_shape):
    """
    Reshapes the convolved matrix to the shape of the image
    :param im: The convolved matrix
    :param res_shape: The expected shape of the result
    :return: The reshaped image
    """
    im = im.transpose((0, 2, 1))
    return im.reshape(im.shape[0], im.shape[1], res_shape[0], res_shape[1])


def conv2D(image, filters, dilation, stride):
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
    imrow = im2rows(image, filters.shape, dilation, stride, dilated_shape, res_shape)
    filtmat = filter2mat(filters)
    res = imrow.dot(filtmat)
    return convolved2im(res, res_shape)


def backward_conv2D(top_grad, image, filters, dilation, stride):
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
    imrow = im2rows(image, filters.shape, dilation, stride, dilated_shape, res_shape)
    filtmat = filter2mat(filters)
    gradmat = top_grad.reshape((top_grad.shape[0], top_grad.shape[1], -1)).transpose((0, 2, 1))
    filt_grad = np.matmul(imrow.transpose((0, 2, 1)), gradmat).sum(axis=0).T.reshape(filter_shape)
    inp_grad_mat = gradmat.dot(filtmat.T)
    inp_grad = backward_im2rows(inp_grad_mat, image, filters.shape, dilation, stride, dilated_shape, res_shape)
    return inp_grad, filt_grad


def maxpool2D(image, pool_shape, dilation=1, stride=None):
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
    imrow = im2rows(image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape, res_shape)
    imrow = imrow.reshape((imrow.shape[0], imrow.shape[1], im_shape[1], -1))
    maxpooled = np.max(imrow, axis=3).transpose((0, 2, 1))
    maxpooled = maxpooled.reshape((maxpooled.shape[0], maxpooled.shape[1], res_shape[0], res_shape[1]))
    cache = np.argmax(imrow, axis=3)
    return maxpooled, cache


def backward_maxpool2D(top_grad, cache, image, pool_shape, dilation=1, stride=None):
    """
    Performs the backward pass on the max-pool operation
    :param top_grad: The grad from the next op
    :param cache: The cache generated in the forward pass
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
    gradrow[i1, i2, i3, cache] = gradmat
    inp_grad = backward_im2rows(gradrow, image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape,
                                res_shape)
    return inp_grad


def meanpool2D(image, pool_shape, dilation=1, stride=None):
    """
    Performs the mean-pooling operation on the image
    :param image: The image to be mean pooled
    :param pool_shape: The shape of the pool filter
    :param dilation: The dilation of the filter
    :param stride: The stride for the filter (defaults to the shape of the pool
    :return: The pooled image
    """
    if stride is None:
        stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = im2rows(image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape, res_shape)
    imrow = imrow.reshape((imrow.shape[0], imrow.shape[1], im_shape[1], -1))
    meanpooled = np.mean(imrow, axis=3).transpose((0, 2, 1))
    meanpooled = meanpooled.reshape((meanpooled.shape[0], meanpooled.shape[1], res_shape[0], res_shape[1]))
    return meanpooled


def backward_meanpool2D(top_grad, image, pool_shape, dilation=1, stride=None):
    """
    Performs the backward pass on the mean-pool operation
    :param top_grad: The grad from the next op
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
    gradmat = top_grad.reshape((top_grad.shape[0], top_grad.shape[1], -1)).transpose((0, 2, 1)) / (
    pool_shape[0] * pool_shape[1])
    gradrow[:, :, :, :] = gradmat[:, :, :, np.newaxis]
    inp_grad = backward_im2rows(gradrow, image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape,
                                res_shape)
    return inp_grad


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


def softmax(x):
    """
    Performs the softmax operation on a 2D tensor
    :param x: The 2D tensor (batch, features)
    :return: The softmaxed tensor
    """
    temp = np.exp(x - x.max(axis=1, keepdims=True))
    res = temp / temp.sum(axis=1, keepdims=True)
    return res


def backward_softmax(top_grad, output):
    """
    Performs the backward pass on the softmax operation
    :param top_grad: The gradient from the next operation
    :param output: The output of this op
    :return: The gradient wrt the input
    """
    left = output[:, :, np.newaxis]
    right = output[:, np.newaxis, :]
    sub = left * np.eye(output.shape[1])
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


def backward_crossentropy(x, y):
    """
    Performs the backward pass through the crossentropy function
    :param x: The input to this op
    :param y: The ground truth
    :return: The gradient wrt the input
    """
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res


def softmax_crossentropy(x, y):
    """
    Calculates the softmax cross-entropy cost
    :param x: The input variable
    :param y: The ground truth (not one-hot)
    :return: The cost and a cache of the softmaxed values
    """
    s = softmax(x)
    return crossentropy(s, y), s


def backward_softmax_crossentropy(cache, y):
    """
    Backward pass through the softmax crossentropy op
    :param cache: The cache generated in the forward pass
    :param y: The ground truth (not one-hot)
    :return: THe gradient wrt the input
    """
    res = cache
    res[np.arange(res.shape[0]), y] -= 1
    return res / cache.shape[0]

def flatten(x):
    """
    Flattens the tensor into a 2D one
    :param x: The tensor to be flattened
    :return: The flattened tensor and a cache
    """
    return x.reshape((x.shape[0], -1)), x.shape

def backward_flatten(top_grad, cache):
    """
    Performs a backward pass on the flatten operation
    :param top_grad: The gradient from the next op
    :param cache: The cache generated during the forward pass
    :return: The gradient wrt the inputs to this op
    """
    return top_grad.reshape(cache)


