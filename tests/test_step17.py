import numpy as np
import pytest
from steps import step17


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step17.as_array(input) == expected

    def test_add(self):
        x0 = step17.Variable(np.array(2))
        x1 = step17.Variable(np.array(3))
        assert step17.add(x0, x1).data == 5


class TestSquare:
    def test_forward(self):
        x = step17.Variable(np.array(2.0))
        y = step17.square(x)
        assert y.data == 4.0

    def test_backward(self):
        x = step17.Variable(np.random.rand(1))
        y = step17.square(x)
        y.backward()
        num_grad = step17.numerical_diff(step17.square, x)
        flg = np.allclose(x.grad, num_grad)
        assert flg


class TestBackward:
    def test_multi_backward(self):
        x = step17.Variable(np.array(2.0))
        a = step17.square(x)
        y = step17.add(step17.square(a), step17.square(a))
        y.backward()
        assert y.data == 32
        assert x.grad == 64


if __name__ == '__main__':
    pytest.main(['-v', __file__])
