
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


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


if __name__ == '__main__':
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    lr = 0.001
    iters = 1000
    for i in range(iters):
        print(x0, x1)
        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()
        x0.data -= x0.grad * lr
        x1.data -= x1.grad * lr
