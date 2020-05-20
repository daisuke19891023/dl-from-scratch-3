import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


data = np.array(2.0)
x = Variable(data)
print(x)
