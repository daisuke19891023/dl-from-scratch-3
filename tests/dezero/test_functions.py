
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
np.random.seed(0)


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


class TestReshape:
    def test_reshape_forward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        assert y.shape == (6,)

    def test_reshape_backward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward(create_graph=False)

        assert x.grad.shape == (2, 3)

    @pytest.mark.parametrize("input, expected", ([([2, 3], (2, 3)), ((2, 3), (2, 3))]))
    def test_reshape_var_method(self, input, expected):
        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = x.reshape(input)
        assert y.shape == expected

    def test_reshape_var_method_tuple(self):
        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = x.reshape(2, 3)
        assert y.shape == (2, 3)


class TestTranspose:
    def test_transpose_forward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        assert y.shape == (3, 2)

    def test_transpose_backward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        y.backward(create_graph=True)
        assert x.grad.shape == (2, 3)

    def test_transpose_var_method(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.transpose()
        assert y.shape == (3, 2)

    def test_transpose_property(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.T
        assert y.shape == (3, 2)


class TestSumTo:
    def test_sum_to_forward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        assert y.data[0] == 11 and y.data[1] == 12 and y.data[2] == 13

    def test_sum_to_backward(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        y.backward()
        assert x1.grad.data == [3]


class TestMeanSquaredError:
    def test_mean_squared_error_forward(self):
        x0 = Variable(np.array([2, 1]))
        x1 = Variable(np.array([1, 2]))
        y = F.mean_squared_error(x0, x1)
        assert y.data == 1
