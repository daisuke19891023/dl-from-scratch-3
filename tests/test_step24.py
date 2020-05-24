import numpy as np
import pytest
from steps import step24 as step


def test_sphere():
    x = step.Variable(np.array(1.0))
    y = step.Variable(np.array(1.0))
    z = step.sphere(x, y)
    z.backward()
    assert x.grad == 2.0 and y.grad == 2.0


def test_metyas():
    x = step.Variable(np.array(1.0))
    y = step.Variable(np.array(1.0))
    z = step.metyas(x, y)
    z.backward()
    assert np.allclose(x.grad, 0.04) and np.allclose(y.grad, 0.04)


def test_goldstein():
    x = step.Variable(np.array(1.0))
    y = step.Variable(np.array(1.0))
    z = step.goldstein(x, y)
    z.backward()
    assert np.allclose(x.grad, -5376.0) and np.allclose(y.grad, 8064.0)
