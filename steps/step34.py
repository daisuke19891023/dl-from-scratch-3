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
    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data.flatten()]

    for i in range(3):
        logs.append(x.grad.data.flatten())
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    # draw graph
    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.show()
