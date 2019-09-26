import os
import numpy as np
import pytest

import deepCR.evaluate as evaluate
from deepCR.model import deepCR


def test_eval():
    mdl = deepCR()
    var = np.zeros((10,24,24))
    tpr, fpr = evaluate.roc(mdl, image=var, mask=var, thresholds=np.linspace(0, 1, 10))
    assert tpr.shape == (10,)
    (tpr, fpr), (tpr1, fpr1) = evaluate.roc(mdl, image=var, mask=var, thresholds=np.linspace(0, 1, 10), dilate=True)
    assert tpr1.shape == (10,)

def test_eval_gen():
    mdl = deepCR()

    # Generate fake data files
    cwd = os.getcwd() + '/'
    # Remove generated files
    if 'temp' in os.listdir(cwd):
        for root, dirs, files in os.walk(cwd + 'temp', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir('temp')
    os.mkdir('temp')
    os.mkdir('temp/image')
    os.mkdir('temp/dark')
    var = np.zeros((2, 24, 24))
    for i in range(6):
        np.save(cwd + 'temp/image/%d.npy' % i, var)
        np.save(cwd + 'temp/dark/%d.npy' % i, var)
    image_list = [cwd + 'temp/image/' + f for f in os.listdir(cwd + 'temp/image')]
    dark_list = [cwd + 'temp/dark/' + f for f in os.listdir(cwd + 'temp/dark')]

    # Evaluate
    tpr, fpr = evaluate.roc(mdl, image=image_list, mask=dark_list, sky=100, thresholds=np.linspace(0, 1, 10))
    assert tpr.shape == (10,)

    # Remove generated files
    for root, dirs, files in os.walk(cwd + 'temp', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir('temp')


def test_eval_gen_lacosmic():
    # Generate fake data files
    cwd = os.getcwd() + '/'
    # Remove generated files
    if 'temp' in os.listdir(cwd):
        for root, dirs, files in os.walk(cwd + 'temp', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir('temp')
    os.mkdir('temp')
    os.mkdir('temp/image')
    os.mkdir('temp/dark')
    var = np.zeros((2, 24, 24))
    for i in range(6):
        np.save(cwd + 'temp/image/%d.npy' % i, var)
        np.save(cwd + 'temp/dark/%d.npy' % i, var)
    image_list = [cwd + 'temp/image/' + f for f in os.listdir(cwd + 'temp/image')]
    dark_list = [cwd + 'temp/dark/' + f for f in os.listdir(cwd + 'temp/dark')]

    # Evaluate
    tpr, fpr = evaluate.roc_lacosmic(image_list, dark_list, sigclip=np.linspace(5,10,10), ignore=None, sky=None,
                                     n_mask=1, seed=1, objlim=2, gain=1, dilate=False, rad=1)
    assert tpr.shape == (10,)

    # Remove generated files
    for root, dirs, files in os.walk(cwd + 'temp', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir('temp')


if __name__ == '__main__':
    test_eval()
    test_eval_gen()
    test_eval_gen_lacosmic()
    