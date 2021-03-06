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
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward()
    print(x.grad)
