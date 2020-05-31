if '__file__' in globals():
    import os
    import sys
    import math
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from dezero import setup_variable
    import dezero.functions as F
    from dezero import optimizers
    from dezero.models import MLP
    from dezero import datasets


setup_variable()

if __name__ == '__main__':
    # hyper paramater
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    # data read
    x, t = datasets.get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(x)
    max_iter = math.ceil(data_size/batch_size)

    for epoch in range(max_epoch):
        # shuffle
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # create mini-batch
            batch_index = index[i * batch_size:(i+1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            # calc grad and update param
            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print("epoch:{}, loss:{}".format(str(epoch + 1), avg_loss))
