from ANN import *
import pickle, gzip

def main():
    Train, Validate, _ = pickle.load(gzip.open("mnist_one_hot.pkl.gz"), encoding="latin1")
    TrainX, TrainY = Train
    TrainX = TrainX.reshape((-1, 1, 28, 28))[:10000]
    TrainY = TrainY.argmax(axis=1)[:10000]
    ValidateX, ValidateY = Validate
    ValidateX = ValidateX.reshape((-1, 1, 28, 28))[:2000]
    ValidateY = ValidateY.argmax(axis=1)[:2000]
    iters = 100
    batch = 100
    num_batches = 100
    num_batches_val = 20
    a = ANN()
    l1 = ConvLayer(10, (3, 3), (1, 1), 2, (1, 28, 28))
    la1 = ReluLayer(l1.output_shape)
    l2 = MaxPoolLayer((2, 2), (2, 2), 1, l1.output_shape)
    l3 = ConvLayer(10, (3, 3), (1, 1), 2, l2.output_shape)
    la3 = ReluLayer(l3.output_shape)
    l4 = MaxPoolLayer((2, 2), (2, 2), 1, l3.output_shape)
    l5 = FlattenLayer(l4.output_shape)
    l6 = FCLayer(30, l5.output_shape)
    la6 = ReluLayer(l6.output_shape)
    l7 = FCLayer(10, l6.output_shape)
    l8 = SoftmaxCELayer(l6.input_shape)
    a.add_layer(l1)
    a.add_layer(la1)
    a.add_layer(l2)
    a.add_layer(l3)
    a.add_layer(la3)
    a.add_layer(l4)
    a.add_layer(l5)
    a.add_layer(l6)
    a.add_layer(la6)
    a.add_layer(l7)
    a.add_layer(l8)
    ord = np.arange(10000)

    for i in range(iters):
        cost = []
        acc = []
        np.random.shuffle(ord)
        TrainX = TrainX[ord]
        TrainY = TrainY[ord]
        for j in range(num_batches):
            cost.append(a.forward(TrainX[j * batch:(j + 1) * batch], TrainY[j * batch:(j + 1) * batch]))
            a.backward()
            a.trainstep()
        for j in range(num_batches_val):
            res = a.run(ValidateX[j * batch:(j + 1) * batch])
            acc.append(np.argmax(res, axis=1) == ValidateY[j * batch:(j + 1) * batch])
        print("Iteration {}".format(i))
        print("Cross-entropy loss: {}".format(np.mean(cost)))
        print("Validation accuracy: {}".format(np.mean(acc)))
        print("-------------------------------------------------------------------")


if __name__ == "__main__":
    main()
