import sys

from tqdm import tqdm

from neural_network.utils import *


def findiff_grad(inp, function, eps=1e-6):
    """
    Approximate the gradient of a function using finite differences
    :param inp: The input at which to evaluate the function
    :param function: The function
    :param eps: The epsilon value to use
    :return: The approximate gradient of the function at the input
    """
    grad = np.zeros_like(inp)
    for i in range(len(inp)):
        inp[i] += eps
        fn_2 = function(inp)
        inp[i] -= 2 * eps
        fn_1 = function(inp)
        grad[i] = (fn_2 - fn_1) / (2 * eps)
        inp[i] += eps
    return grad


def check_conv2d():
    """
    Run tests for the conv2d gradient implementation.
    """
    input_shapes = [(1, 1, 28, 28), (2, 5, 27, 27), (3, 15, 31, 31)]
    filter_shapes = [(32, 1, 4, 4), (11, 5, 19, 19), (32, 15, 3, 3)]
    strides = [(1, 1), (2, 2), (3, 3)]
    dilations = [2, 1, 3]
    n_tests = len(input_shapes)
    tqdm.write('Running conv2d tests...')
    for test in tqdm(zip(input_shapes, filter_shapes, strides, dilations), total=n_tests, file=sys.stdout):
        inp_shape, filt_shape, stride, dilation = test
        image = np.random.random(inp_shape) * 2 - 1
        filter_ = np.random.random(filt_shape) * 2 - 1
        convolved = conv2d(image, filter_, dilation, stride)
        conv_grad = backward_tensor_sum(convolved)
        image_grad, filter_grad = backward_conv2d(conv_grad, image, filter_, dilation, stride)
        findiff_image_grad = findiff_grad(image.flatten(), lambda x: tensor_sum(conv2d(x.reshape(image.shape),
                                                                                       filter_, dilation, stride)))
        findiff_filt_grad = findiff_grad(filter_.flatten(), lambda x: tensor_sum(conv2d(image,
                                                                                        x.reshape(filter_.shape),
                                                                                        dilation, stride)))
        assert np.allclose(filter_grad.flatten(), findiff_filt_grad, atol=1e-6), f'Filter grad wrong for {test}'
        assert np.allclose(image_grad.flatten(), findiff_image_grad, atol=1e-6), f'Input grad wrong for {test}'
    tqdm.write('All tests passed\n------------------------------------------------------')


def check_transposed_conv2d():
    """
    Run tests for the transposed convolution gradient implementation.
    """
    input_shapes = [(1, 1, 7, 7), (2, 5, 3, 3), (3, 15, 1, 1)]
    filter_shapes = [(1, 13, 4, 4), (5, 8, 1, 1), (15, 20, 5, 5)]
    strides = [(1, 1), (2, 2), (3, 3)]
    dilations = [2, 1, 3]
    n_tests = len(input_shapes)
    tqdm.write('Running transposed_conv2d tests...')
    for test in tqdm(zip(input_shapes, filter_shapes, strides, dilations), total=n_tests, file=sys.stdout):
        inp_shape, filt_shape, stride, dilation = test
        image = np.random.random(inp_shape) * 2 - 1
        filter_ = np.random.random(filt_shape) * 2 - 1
        t_convolved = transposed_conv2d(image, filter_, dilation, stride)
        t_convolved_grad = backward_tensor_sum(t_convolved)
        image_grad, filter_grad = backward_transposed_conv2d(t_convolved_grad, image, filter_, dilation, stride)
        findiff_image_grad = findiff_grad(image.flatten(), lambda x: tensor_sum(
            transposed_conv2d(x.reshape(image.shape), filter_, dilation, stride))
            )
        findiff_filt_grad = findiff_grad(filter_.flatten(), lambda x: tensor_sum(
            transposed_conv2d(image, x.reshape(filter_.shape), dilation, stride))
            )
        assert np.allclose(filter_grad.flatten(), findiff_filt_grad, atol=1e-6), f'Filter grad wrong for {test}'
        assert np.allclose(image_grad.flatten(), findiff_image_grad, atol=1e-6), f'Input grad wrong for {test}'
    tqdm.write('All tests passed\n------------------------------------------------------')


def main():
    check_conv2d()
    check_transposed_conv2d()


if __name__ == '__main__':
    main()
