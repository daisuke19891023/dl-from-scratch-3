
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from dezero import Function
    from dezero import Variable
    from dezero import setup_variable
    from dezero.utils import plot_dot_graph
    import math
setup_variable()


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    iters = 10
    for i in range(iters):
        print(i, x)
        y = f(x)
        x.cleargrad()
        y.backward()
        x.data -= x.grad / gx2(x.data)
