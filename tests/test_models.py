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
    import dezero.models as M
    import pytest
setup_variable()
np.random.seed(0)
lr = 0.2
iters = 10001


class TestMLP:
    def test_mlp(self):
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        max_iters = 100

        model = M.MLP((10, 20, 30, 40, 1))
        for i in range(max_iters):
            y_pred = model(x)
            loss = F.mean_squared_error(y, y_pred)

            model.cleargrads()
            loss.backward()
            for p in model.params():
                assert p.grad.data is not None
