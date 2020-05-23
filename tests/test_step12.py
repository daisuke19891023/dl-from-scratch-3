import numpy as np
import pytest
from steps import step12


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert step12.as_array(input) == expected

    def test_add(self):
        x0 = step12.Variable(np.array(2))
        x1 = step12.Variable(np.array(3))
        assert step12.add(x0, x1).data == 5


if __name__ == '__main__':
    pytest.main(['-v', __file__])
