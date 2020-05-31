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
    from dezero import DataLoader
    import dezero

setup_variable()


# hyper paramater
max_epoch = 3
batch_size = 100
hidden_size = 1000

# data read
train_set = datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)


if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)

    print('epoch:{}'.format(epoch + 1))
    print('loss: {:.4f}'.format(sum_loss / len(train_set)))

model.save_weights('my_mlp.npz')
