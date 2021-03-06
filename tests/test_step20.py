import numpy as np
import pytest
from steps import step20 as step


class TestVariable:
    def test_name(self):
        x = step.Variable(np.array(1), "test")
        assert x.name == "test"

    @pytest.mark.parametrize("input, expected", [(step.Variable(np.array(1)), 'int32'), (step.Variable(np.array([1.0, 2.0, 3.0])),  'float64'), (step.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])),  'float64')])
    def test_dtype(self, input, expected):
        assert input.dtype == expected

    @pytest.mark.parametrize("input, expected", [(step.Variable(np.array(1.0)), 0), (step.Variable(np.array([1.0, 2.0, 3.0])), 1), (step.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 2)])
    def test_ndim(self, input, expected):
        assert input.ndim == expected

    @pytest.mark.parametrize("input, expected", [(step.Variable(np.array(1.0)), 1), (step.Variable(np.array([1.0, 2.0, 3.0])), 3), (step.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 4)])
    def test_size(self, input, expected):
        assert input.size == expected

    @pytest.mark.parametrize("input, expected", [(step.Variable(np.array(1.0)), 1), (step.Variable(np.array([1.0, 2.0, 3.0])), 3), (step.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 2)])
    def test_len(self, input, expected):
        assert len(input) == expected

    @pytest.mark.parametrize("input, expected", [(step.Variable(np.array(1.0)), ()), (step.Variable(np.array([1.0, 2.0, 3.0])), (3,)), (step.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), (2, 2))])
    def test_shape(self, input, expected):
        assert input.shape == expected

    @pytest.mark.parametrize("input, expected", [(step.Variable(np.array(1.0)), "variable(1.0)\n"), (step.Variable(np.array([1.0, 2.0, 3.0])),  "variable([1. 2. 3.])\n"), (step.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])),  "variable([[1. 2.]\n          [2. 3.]])\n")])
    def test_shape(self, input, expected, capsys):
        print(input)
        captured = capsys.readouterr()
        assert captured.out == expected


@pytest.fixture(scope="function", autouse=True)
def culc_object():
    yield step.Variable(np.array(2)), step.Variable(np.array(3))


class TestAdd:
    def test_add(self, culc_object):
        assert step.add(culc_object[0], culc_object[1]).data == 5

    def test_add_backward(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = step.add(x0, x1)
        y.backward()
        assert x0.grad == 1 and x1.grad == 1

    def test_add_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert (x0 + x1).data == 5

    def test_add_backward_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = x0 + x1
        y.backward()
        assert x0.grad == 1 and x1.grad == 1


class TestMul:
    def test_mul(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert step.mul(x0, x1).data == 6

    def test_mul_backward(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = step.mul(x0, x1)
        y.backward()
        assert x0.grad == 3 and x1.grad == 2

    def test_mul_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert (x0 * x1).data == 6

    def test_mul_backward_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = step.mul(x0, x1)
        y.backward()
        assert x0.grad == 3 and x1.grad == 2


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step.as_array(input) == expected

    def test_no_grad(self):
        with step.no_grad():
            assert not step.Config.enable_backprop


class TestSquare:
    def test_forward(self):
        x = step.Variable(np.array(2.0))
        y = step.square(x)
        assert y.data == 4.0

    def test_backward(self):
        x = step.Variable(np.random.rand(1))
        y = step.square(x)
        y.backward()
        num_grad = step.numerical_diff(step.square, x)
        flg = np.allclose(x.grad, num_grad)
        assert flg


class TestBackward:
    def test_multi_backward(self):
        x = step.Variable(np.array(2.0))
        a = step.square(x)
        y = step.add(step.square(a), step.square(a))
        y.backward()
        assert y.data == 32
        assert x.grad == 64


if __name__ == '__main__':
    pytest.main(['-v', __file__])
