if '__file__' in globals():
    import os
    import sys
    import math
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy as np
    from PIL import Image
    from dezero import Variable
    from dezero import setup_variable
    from dezero.models import VGG16
    import dezero
setup_variable()
if __name__ == '__main__':
    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/' \
        'raw/images/zebra.jpg'
    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)
    x = VGG16.preprocess(img)
    x = x[np.newaxis]

    model = VGG16(pretrained=True)
    with dezero.test_mode():
        y = model(x)
    predoct_id = np.argmax(y.data)
    model.plot(x, to_file='vgg.pdf')
    labels = dezero.datasets.ImageNet.labels()
    print(labels[predoct_id])
