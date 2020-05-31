if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from dezero import Variable, Model, as_variable
    from dezero import setup_variable
    from dezero.utils import plot_dot_graph
    import dezero.functions as F
    from dezero import optimizers
    from dezero.models import MLP


setup_variable()


def softmaxld(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


if __name__ == '__main__':
    x = Variable(np.array([[0.2, -0.4]]))
    model = MLP((10, 3))
    y = model(x)
    p = softmaxld(y)
    print(y)
    print(p)

    a = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0])
    z = model(a)
    loss = F.softmax_cross_entropy_simple(z, t)
    print(loss)
