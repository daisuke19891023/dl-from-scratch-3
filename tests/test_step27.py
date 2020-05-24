import numpy as np
import pytest
from steps import step27 as step


class TestSin:
    def test_sin_forward(self):
        x = step.Variable(np.array(np.pi / 4))
        y = step.sin(x)
        assert np.allclose(y.data, 0.7071)

    def test_sin_backward(self):
        x = step.Variable(np.array(np.pi / 4))
        y = step.sin(x)
        y.backward()
        assert np.allclose(x.grad, 0.7071)

    def test_ny_sin_forward(self):
        x = step.Variable(np.array(np.pi / 4))
        y = step.my_sin(x)
        assert np.allclose(y.data, 0.7071)

    def test_sin_backward(self):
        x = step.Variable(np.array(np.pi / 4))
        y = step.my_sin(x)
        y.backward()
        assert np.allclose(x.grad, 0.7071)
