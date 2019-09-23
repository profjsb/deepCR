import os
import numpy as np
import pytest

from deepCR.dataset import dataset, DatasetSim


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


def test_DatasetSim():
    cwd = os.getcwd() + '/'

    os.mkdir('temp')
    os.mkdir('temp/image')
    os.mkdir('temp/dark')

    var = np.zeros((2, 64, 64))
    for i in range(10):
        np.save(cwd + 'temp/image/%d.npy' % i, var)
        np.save(cwd + 'temp/dark/%d.npy' % i, var)
    image_list = [cwd + 'temp/image/' + f for f in os.listdir(cwd + 'temp/image')]
    dark_list = [cwd + 'temp/dark/' + f for f in os.listdir(cwd + 'temp/dark')]

    data = DatasetSim(image=image_list, cr=dark_list, sky=100, part='train', aug_sky=[1, 1], aug_img=[0.5, 2], f_val=0.1)
    data0 = data[0]
    assert len(data) == 9
    assert data0[0].shape == (64, 64)
    assert data0[1].shape == (64, 64)
    assert data0[2].shape == (64, 64)

    for root, dirs, files in os.walk(cwd + 'temp', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir('temp')


if __name__ == '__main__':
    test_dataset()
    test_DatasetSim()
