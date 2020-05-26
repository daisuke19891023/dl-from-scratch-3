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
    import dezero.layers as L
    import pytest

setup_variable()
np.random.seed(0)
lr = 0.2
iters = 10001


@pytest.fixture(scope="function", autouse=True)
def linear_object():
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    yield x, y


def predict(x, l1, l2):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


class TestLinear:
    def test_linear_forward(self, linear_object):
        l1 = L.Linear(10)
        l2 = L.Linear(1)
        y_pred = predict(linear_object[0], l1, l2)
        loss = F.mean_squared_error(linear_object[1], y_pred)

        assert np.allclose(loss.data, 0.81651785)

    def test_linear_backward(self, linear_object):
        l1 = L.Linear(10)
        l2 = L.Linear(1)
        y_pred = predict(linear_object[0], l1, l2)
        loss = F.mean_squared_error(linear_object[1], y_pred)
        l1.cleargrads()
        l2.cleargrads()
        loss.backward()
        for l in [l1, l2]:
            for p in l.params():
                assert p.grad.data is not None
