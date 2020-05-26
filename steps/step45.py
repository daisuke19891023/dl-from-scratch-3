if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    import matplotlib.pyplot as plt
    from dezero import Variable, Model
    from dezero import setup_variable
    from dezero.utils import plot_dot_graph
    import dezero.functions as F
    import dezero.layers as L

setup_variable()


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


def my_func1():
    x = Variable(np.random.randn(5, 10), name='x')
    model = TwoLayerNet(100, 10)
    model.plot(x)


if __name__ == '__main__':
    np.random.seed(0)
    # my_func1()
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    lr = 0.2
    max_iters = 10001
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)
    for i in range(max_iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()
        for p in model.params():
            p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss)
