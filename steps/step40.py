if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    import matplotlib.pyplot as plt
    from dezero import Variable
    from dezero import setup_variable
    from dezero.utils import plot_dot_graph
    from dezero.utils import sum_to
setup_variable()

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = sum_to(x, (1, 1))
print(y)
