import numpy as np
import pytest
from steps import step18


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step18.as_array(input) == expected

    def test_add(self):
        x0 = step18.Variable(np.array(2))
        x1 = step18.Variable(np.array(3))
        assert step18.add(x0, x1).data == 5

    def test_no_grad(self):
        with step18.no_grad():
            assert not step18.Config.enable_backprop


class TestSquare:
    def test_forward(self):
        x = step18.Variable(np.array(2.0))
        y = step18.square(x)
        assert y.data == 4.0

    def test_backward(self):
        x = step18.Variable(np.random.rand(1))
        y = step18.square(x)
        y.backward()
        num_grad = step18.numerical_diff(step18.square, x)
        flg = np.allclose(x.grad, num_grad)
        assert flg


class TestBackward:
    def test_multi_backward(self):
        x = step18.Variable(np.array(2.0))
        a = step18.square(x)
        y = step18.add(step18.square(a), step18.square(a))
        y.backward()
        assert y.data == 32
        assert x.grad == 64


if __name__ == '__main__':
    pytest.main(['-v', __file__])
