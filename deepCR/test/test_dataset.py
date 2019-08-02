import numpy as np
import pytest

from deepCR.dataset import dataset


def test_dataset():
    inputs = np.random.rand(10,32,32); sky = np.random.rand(10)
    data = dataset(image=inputs, mask=inputs, ignore=inputs, sky=sky, part='train', aug_sky=[1, 1], f_val=0.1)
    data0 = data[0]
    assert len(data) == 9
    assert (data0[0] == (inputs[0] + sky[0])).all()
    assert (data0[1] == inputs[0]).all()
    assert (data0[2] == inputs[0]).all()

    data = dataset(image=inputs, mask=inputs)
    assert len(data) == 10


if __name__ == '__main__':
    test_dataset()
