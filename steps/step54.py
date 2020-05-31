if '__file__' in globals():
    import os
    import sys
    import math
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from dezero import setup_variable
    import dezero.functions as F
    from dezero import test_mode

setup_variable()

if __name__ == '__main__':
    x = np.ones(5)
    print(x)

    # train
    y = F.dropout(x)
    print(y)

    # test
    with test_mode():
        y = F.dropout(x)
        print(y)
