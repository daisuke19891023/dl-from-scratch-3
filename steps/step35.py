if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    import matplotlib.pyplot as plt
    from dezero import Variable
    from dezero import setup_variable
    from dezero.utils import plot_dot_graph
    import dezero.functions as F
setup_variable()


if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)
    iters = 8

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        # draw graph
        gx.name = 'gx'+str(i + 1)
        plot_dot_graph(gx, verbose=False, to_file='tanh{}.png'.format(str(i)))
