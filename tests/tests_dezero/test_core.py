
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
    import numpy as np
    import pytest

setup_variable()


class TestVariable:
    def test_name(self):
        x = Variable(np.array(1), "test")
        assert x.name == "test"

    @pytest.mark.parametrize("input, expected", [(Variable(np.array(1)), 'int32'), (Variable(np.array([1.0, 2.0, 3.0])),  'float64'), (Variable(np.array([[1.0, 2.0], [2.0, 3.0]])),  'float64')])
    def test_dtype(self, input, expected):
        assert input.dtype == expected

    @pytest.mark.parametrize("input, expected", [(Variable(np.array(1.0)), 0), (Variable(np.array([1.0, 2.0, 3.0])), 1), (Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 2)])
    def test_ndim(self, input, expected):
        assert input.ndim == expected

    @pytest.mark.parametrize("input, expected", [(Variable(np.array(1.0)), 1), (Variable(np.array([1.0, 2.0, 3.0])), 3), (Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 4)])
    def test_size(self, input, expected):
        assert input.size == expected

    @pytest.mark.parametrize("input, expected", [(Variable(np.array(1.0)), 1), (Variable(np.array([1.0, 2.0, 3.0])), 3), (Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), 2)])
    def test_len(self, input, expected):
        assert len(input) == expected

    @pytest.mark.parametrize("input, expected", [(Variable(np.array(1.0)), ()), (Variable(np.array([1.0, 2.0, 3.0])), (3,)), (Variable(np.array([[1.0, 2.0], [2.0, 3.0]])), (2, 2))])
    def test_shape(self, input, expected):
        assert input.shape == expected

    @pytest.mark.parametrize("input, expected", [(Variable(np.array(1.0)), "variable(1.0)\n"), (Variable(np.array([1.0, 2.0, 3.0])),  "variable([1. 2. 3.])\n"), (Variable(np.array([[1.0, 2.0], [2.0, 3.0]])),  "variable([[1. 2.]\n          [2. 3.]])\n")])
    def test_shape(self, input, expected, capsys):
        print(input)
        captured = capsys.readouterr()
        assert captured.out == expected

    @pytest.mark.parametrize("input, expected", [(np.array(1.0), Variable(np.array(1.0))), (Variable(np.array(1.0)), Variable(np.array(1.0)))])
    def test_as_variable(self, input, expected):
        assert type(as_variable(input)) == type(expected)


@pytest.fixture(scope="function", autouse=True)
def culc_object():
    yield Variable(np.array(2)), Variable(np.array(3))


class TestNeg:
    def test_neg(self):
        x = Variable(np.array(2.0))
        assert neg(x).data == -2.0

    def test_neg_backward(self):
        x = Variable(np.array(2.0))
        y = -x
        y.backward()
        assert x.grad.data == -1

    def test_neg_overload(self):
        x = Variable(np.array(2.0))
        assert (-x).data == -2.0


class TestAdd:
    def test_add(self, culc_object):
        assert add(culc_object[0], culc_object[1]).data == 5

    def test_add_backward(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = add(x0, x1)
        y.backward()
        assert x0.grad.data == 1 and x1.grad.data == 1

    def test_add_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert (x0 + x1).data == 5

    def test_add_backward_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = x0 + x1
        y.backward()
        assert x0.grad.data == 1 and x1.grad.data == 1

    def test_calc_multi_type(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        assert y.data == Variable(np.array(5.0)).data

    def test_calc_multi_type_left(self):
        x = Variable(np.array(2.0))
        y = x + 3
        assert y.data == Variable(np.array(5.0)).data

    def test_calc_multi_type_right(self):
        x = Variable(np.array(2.0))
        y = 3.0 + x
        assert y.data == Variable(np.array(5.0)).data

    def test_calc_multi_type_ndarray_right(self):
        x = Variable(np.array(2.0))
        y = np.array(3.0) + x
        assert y.data == Variable(np.array(5.0)).data


class TestSub:
    def test_sub(self, culc_object):
        assert sub(culc_object[0], culc_object[1]).data == -1

    def test_sub_backward(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = sub(x0, x1)
        y.backward()
        assert x0.grad.data == 1 and x1.grad.data == -1

    def test_sub_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert (x0 - x1).data == -1

    def test_sub_backward_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = x0 - x1
        y.backward()
        assert x0.grad.data == 1 and x1.grad.data == -1


class TestMul:
    def test_mul(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert mul(x0, x1).data == 6

    def test_mul_backward(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = mul(x0, x1)
        y.backward()
        assert x0.grad.data == 3 and x1.grad.data == 2

    def test_mul_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert (x0 * x1).data == 6

    def test_mul_backward_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = mul(x0, x1)
        y.backward()
        assert x0.grad.data == 3 and x1.grad.data == 2


class TestDiv:
    def test_div(self, culc_object):
        assert div(culc_object[0], culc_object[1]).data == 2 / 3

    def test_div_backward(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = div(x0, x1)
        y.backward()
        assert x0.grad.data == 1 / 3 and x1.grad.data == -2 / 9

    def test_div_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        assert (x0 / x1).data == 2 / 3

    def test_div_backward_overload(self, culc_object):
        x0 = culc_object[0]
        x1 = culc_object[1]
        y = x0 / x1
        y.backward()
        assert x0.grad.data == 1 / 3 and x1.grad.data == -2 / 9


class TestPow:
    def test_pow(self):
        x = Variable(np.array(2.0))
        assert pow(x, 3).data == 8

    def test_pow_backward(self):
        x = Variable(np.array(2.0))
        y = pow(x, 3)
        y.backward()
        assert x.grad.data == 12

    def test_pow_overload(self):
        x = Variable(np.array(2.0))
        assert (x ** 3).data == 8

    def test_pow_backward_overload(self):
        x = Variable(np.array(2.0))
        y = x ** 3
        y.backward()
        assert x.grad.data == 12


class TestUtils:
    @pytest.mark.parametrize("input, expected", [(np.array(1.0), np.array(1.0)), (np.array(1.0, np.float64), np.array(1.0))])
    def test_as_array(self, input, expected):
        assert as_array(input) == expected

    def test_no_grad(self):
        with no_grad():
            assert not Config.enable_backprop


if __name__ == '__main__':
    pytest.main(['-v', __file__])
