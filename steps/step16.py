import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from nptyping import Array


class Variable:
    def __init__(self, data: Union[float, Array[float, ..., ...]]) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self) -> None:

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            print("funcs", funcs)
            print("seen", seen_set)
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

    def cleargrad(self) -> None:
        self.grad = None


class Function:
    def __call__(self, *inputs: Variable) -> Union[List[Variable], Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: Union[float, Array[float, ..., ...]]) -> float:
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


def square(x: Variable) -> Union[List[Variable], Variable]:
    return Square()(x)


def as_array(x: Union[float, Array[float, ..., ...]]) -> Array[float, ..., ...]:
    if np.isscalar(x):
        return np.array(x)
    return x


def numerical_diff(f: Function, x: Variable, eps=1e-4) -> Optional[float]:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    if isinstance(y0, Variable) and isinstance(y1, Variable):
        return (y1.data - y0.data) / (2 * eps)
    else:
        return None


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    a = square(x)
    if isinstance(a, Variable):
        y = add(square(a), square(a))
        y.backward()

        print(y.data)
        print(x.grad)
