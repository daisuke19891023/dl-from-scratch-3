import numpy as np
import pytest
from steps import step13


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step13.as_array(input) == expected

    def test_add(self):
        x0 = step13.Variable(np.array(2))
        x1 = step13.Variable(np.array(3))
        assert step13.add(x0, x1).data == 5


class TestSquare:
    def test_forward(self):
        x = step13.Variable(np.array(2.0))
        y = step13.square(x)
        assert y.data == 4.0

    def test_backward(self):
        x = step13.Variable(np.random.rand(1))
        y = step13.square(x)
        y.backward()
        num_grad = step13.numerical_diff(step13.square, x)
        flg = np.allclose(x.grad, num_grad)
        assert flg


if __name__ == '__main__':
    pytest.main(['-v', __file__])
