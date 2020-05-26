
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dezero import Variable
    from dezero import Function
    from dezero import using_config
    from dezero import no_grad
    from dezero import as_array
    from dezero import as_variable
    from dezero import setup_variable
    from dezero.core import Config, add, mul, neg, sub, rsub, div, rdiv, pow
    import dezero.functions as F
    import numpy as np
    import pytest

setup_variable()


class TestSin:
    def test_sin_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = F.sin(x)
        assert np.allclose(y.data, 0.7071)

    def test_sin_backward_once(self):
        x = Variable(np.array(np.pi / 4))
        y = F.sin(x)
        y.backward()
        result = x.grad.data
        assert np.allclose(result, 0.70710678)

    def test_sin_backward_twice(self):
        x = Variable(np.array(1.0))
        y = F.sin(x)
        y.backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=False)
        result = x.grad.data
        assert np.allclose(result, -0.84147098)


class TestCos:
    def test_cos_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = F.cos(x)
        assert np.allclose(y.data, 0.70710678)

    def test_cos_backward_once(self):
        x = Variable(np.array(np.pi / 4))
        y = F.cos(x)
        y.backward()
        result = x.grad.data
        assert np.allclose(result, -0.7071)

    def test_cos_backward_twice(self):
        x = Variable(np.array(1.0))
        y = F.cos(x)
        y.backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=False)
        result = x.grad.data
        assert np.allclose(result, -0.54030231)


class TestTanh:
    def test_tanh_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = F.tanh(x)
        assert np.allclose(y.data, 0.6557942)

    def test_tanh_backward_once(self):
        x = Variable(np.array(np.pi / 4))
        y = F.tanh(x)
        y.backward()
        result = x.grad.data
        assert np.allclose(result, 0.56993396)

    def test_tanh_backward_twice(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        y.backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=False)
        result = x.grad.data
        assert np.allclose(result, -0.63970001)
