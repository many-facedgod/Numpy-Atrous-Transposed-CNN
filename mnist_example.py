import pickle
import gzip
import sys

from tqdm import tqdm, trange

from neural_network.network import *


def load_data():
    """Load the MNIST data and normalize it."""
    (trainx, trainy), (valx, valy), (testx, testy) = pickle.load(gzip.open("data/mnist_one_hot.pkl.gz"),
                                                                 encoding="latin1")
    trainy = np.argmax(trainy, axis=1)
    valy = np.argmax(valy, axis=1)
    testy = np.argmax(testy, axis=1)
    trainx = trainx * 2 - 1
    valx = valx * 2 - 1
    testx = testx * 2 - 1
    return (trainx.reshape(-1, 1, 28, 28), trainy), (valx.reshape(-1, 1, 28, 28), valy), (testx.reshape(-1, 1, 28, 28),
                                                                                          testy)


def train_classifier(data, n_iters=3, batch_size=100):
    """
    Train a CNN classifier on the data
    :param data: The MNIST data loaded
    :param n_iters: The number of iterations to train for
    :param batch_size: The batch size to use
    """
    tqdm.write(f'Training a dilated CNN classifier for {n_iters} iterations.')
    (trainx, trainy), (valx, valy), (testx, testy) = data
    train_size, val_size, test_size = trainx.shape[0], valx.shape[0], testx.shape[0]
    train_batches = (train_size - 1) // batch_size + 1
    val_batches = (val_size - 1) // batch_size + 1
    test_batches = (test_size - 1) // batch_size + 1

    model = Network()
    model.add_layer(ConvLayer(10, (3, 3), (1, 1), 2)) \
        .add_layer(ReluLayer()) \
        .add_layer(Pad2DLayer((2, 2))) \
        .add_layer(ConvLayer(10, (3, 3), (1, 1), 2)) \
        .add_layer(ReluLayer()) \
        .add_layer(Pool2DLayer((2, 2))) \
        .add_layer(ConvLayer(10, (3, 3), (1, 1), 2)) \
        .add_layer(ReluLayer()) \
        .add_layer(Pool2DLayer((2, 2))) \
        .add_layer(FlattenLayer()) \
        .add_layer(FCLayer(32)) \
        .add_layer(ReluLayer()) \
        .add_layer(FCLayer(10)) \
        .add_layer(SoftmaxCELayer())
    for i in range(1, n_iters + 1):
        train_order = np.random.permutation(train_size)
        bar = trange(train_batches, file=sys.stdout)
        for j in bar:
            cost = model.forward(trainx[train_order[j * batch_size: (j + 1) * batch_size]],
                                 trainy[train_order[j * batch_size: (j + 1) * batch_size]])
            bar.set_description(f'Curr loss: {cost}')
            model.backward()
            model.adam_trainstep()
        correct = []
        for j in range(val_batches):
            res = model.run(valx[j * batch_size:(j + 1) * batch_size])
            correct.append(np.argmax(res, axis=1) == valy[j * batch_size:(j + 1) * batch_size])
        tqdm.write(f'Validation accuracy: {np.mean(correct)}')
        tqdm.write('-------------------------------------------------------')

    correct = []
    for i in range(test_batches):
        res = model.run(testx[i * batch_size:(i + 1) * batch_size])
        correct.append(np.argmax(res, axis=1) == testy[i * batch_size:(i + 1) * batch_size])
    tqdm.write(f'Test accuracy: {np.mean(correct)}')
    tqdm.write('-------------------------------------------------------')


def train_autoencoder(data, n_iters=10, batch_size=100):
    """
    Train a convolution-transposed convolution based autoencoder
    :param data: The loaded MNIST data
    :param n_iters: The number of iterations
    :param batch_size: The batch size to use
    """
    tqdm.write(f'Training a fully-convolutional autoencoder for {n_iters} iterations.')
    (trainx, trainy), (valx, valy), (testx, testy) = data
    train_size, val_size, test_size = trainx.shape[0], valx.shape[0], testx.shape[0]
    train_batches = (train_size - 1) // batch_size + 1
    val_batches = (val_size - 1) // batch_size + 1
    test_batches = (test_size - 1) // batch_size + 1

    model = Network()
    model.add_layer(ConvLayer(10, (2, 2), (2, 2), 1)) \
        .add_layer(ConvLayer(10, (2, 2), (2, 2), 1)) \
        .add_layer(ConvLayer(15, (1, 1), (2, 2), 1)) \
        .add_layer(TransposedConvLayer(10, (1, 1), (2, 2), 1)) \
        .add_layer(TransposedConvLayer(10, (2, 2), (2, 2), 1)) \
        .add_layer(TransposedConvLayer(1, (2, 2), (2, 2), 1)) \
        .add_layer(SSELayer())
    for i in range(1, n_iters + 1):
        train_order = np.random.permutation(train_size)
        bar = trange(train_batches, file=sys.stdout)
        for j in bar:
            cost = model.forward(trainx[train_order[j * batch_size: (j + 1) * batch_size]],
                                 trainx[train_order[j * batch_size: (j + 1) * batch_size]])
            bar.set_description(f'Curr squared error: {cost}')
            model.backward()
            model.adam_trainstep()
        errors = []
        for j in range(val_batches):
            errors.append(model.forward(valx[j * batch_size:(j + 1) * batch_size],
                                        valx[j * batch_size:(j + 1) * batch_size]))
        tqdm.write(f'Validation squared error: {np.mean(errors)}')
        tqdm.write('-------------------------------------------------------')

    errors = []
    for i in range(test_batches):
        errors.append(model.forward(testx[i * batch_size:(i + 1) * batch_size],
                                    testx[i * batch_size:(i + 1) * batch_size]))
    tqdm.write(f'Test squared error: {np.mean(errors)}')
    tqdm.write('-------------------------------------------------------')


def main():
    data = load_data()
    train_classifier(data)
    train_autoencoder(data)


if __name__ == "__main__":
    main()
