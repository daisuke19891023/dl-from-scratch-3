import numpy as np
import pytest
from steps import step16


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step16.as_array(input) == expected

    def test_add(self):
        x0 = step16.Variable(np.array(2))
        x1 = step16.Variable(np.array(3))
        assert step16.add(x0, x1).data == 5


class TestSquare:
    def test_forward(self):
        x = step16.Variable(np.array(2.0))
        y = step16.square(x)
        assert y.data == 4.0

    def test_backward(self):
        x = step16.Variable(np.random.rand(1))
        y = step16.square(x)
        y.backward()
        num_grad = step16.numerical_diff(step16.square, x)
        flg = np.allclose(x.grad, num_grad)
        assert flg


class TestBackward:
    def test_multi_backward(self):
        x = step16.Variable(np.array(2.0))
        a = step16.square(x)
        y = step16.add(step16.square(a), step16.square(a))
        y.backward()
        assert y.data == 32
        assert x.grad == 64


if __name__ == '__main__':
    pytest.main(['-v', __file__])
