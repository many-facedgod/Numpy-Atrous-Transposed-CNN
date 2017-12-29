# Numpy GEMM-based Atrous/Dilated CNN

A pure numpy-GEMM based implementation of the dilated/atrous CNNs. The methods in the Utils file can be used as a reference for low-level implementation of general 2D convolutions. The pooling operations also support the same parameters as the convolution operator (including dilation, although I cannot think of a case where it will be useful). To counter the python overhead, the Numba JIT compiler has been used. The BLAS GEMM, of course, is accessed using the numpy methods.

### Dependencies:
- Numpy
- Numba (recommended)

### Running the MNIST network:
Just run the MNIST_test.py file. The network architecture is for demonstration purposes only and is very shallow. 

### Todo: 
 - Transposed convolutions

### References:
- Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." arXiv preprint arXiv:1606.00915 (2016).

