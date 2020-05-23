import numpy as np
import pytest
from steps import step19


class TestVariable:
    def test_name(self):
        x = step19.Variable(np.array(1), "test")
        assert x.name == "test"

    @pytest.mark.parametrize("input, expected", [(step19.Variable(np.array(1)), 'int32'), (step19.Variable(np.array([1.0, 2.0, 3.0])),  'float64'), (step19.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])),  'float64')])
    def test_dtype(self, input, expected):
        assert input.dtype == expected

    @pytest.mark.parametrize("input, expected", [(step19.Variable(np.array(1.0)), 0), (step19.Variable(np.array([1.0, 2.0, 3.0])), 1), (step19.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 2)])
    def test_ndim(self, input, expected):
        assert input.ndim == expected

    @pytest.mark.parametrize("input, expected", [(step19.Variable(np.array(1.0)), 1), (step19.Variable(np.array([1.0, 2.0, 3.0])), 3), (step19.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 4)])
    def test_size(self, input, expected):
        assert input.size == expected

    @pytest.mark.parametrize("input, expected", [(step19.Variable(np.array(1.0)), 1), (step19.Variable(np.array([1.0, 2.0, 3.0])), 3), (step19.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 2)])
    def test_len(self, input, expected):
        assert len(input) == expected

    @pytest.mark.parametrize("input, expected", [(step19.Variable(np.array(1.0)), ()), (step19.Variable(np.array([1.0, 2.0, 3.0])), (3,)), (step19.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), (2, 2))])
    def test_shape(self, input, expected):
        assert input.shape == expected

    @pytest.mark.parametrize("input, expected", [(step19.Variable(np.array(1.0)), "variable(1.0)\n"), (step19.Variable(np.array([1.0, 2.0, 3.0])),  "variable([1. 2. 3.])\n"), (step19.Variable(np.array([[1.0, 2.0], [2.0, 3.0]])),  "variable([[1. 2.]\n          [2. 3.]])\n")])
    def test_shape(self, input, expected, capsys):
        print(input)
        captured = capsys.readouterr()
        assert captured.out == expected


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step19.as_array(input) == expected

    def test_add(self):
        x0 = step19.Variable(np.array(2))
        x1 = step19.Variable(np.array(3))
        assert step19.add(x0, x1).data == 5

    def test_no_grad(self):
        with step19.no_grad():
            assert not step19.Config.enable_backprop


class TestSquare:
    def test_forward(self):
        x = step19.Variable(np.array(2.0))
        y = step19.square(x)
        assert y.data == 4.0

    def test_backward(self):
        x = step19.Variable(np.random.rand(1))
        y = step19.square(x)
        y.backward()
        num_grad = step19.numerical_diff(step19.square, x)
        flg = np.allclose(x.grad, num_grad)
        assert flg


class TestBackward:
    def test_multi_backward(self):
        x = step19.Variable(np.array(2.0))
        a = step19.square(x)
        y = step19.add(step19.square(a), step19.square(a))
        y.backward()
        assert y.data == 32
        assert x.grad == 64


if __name__ == '__main__':
    pytest.main(['-v', __file__])
