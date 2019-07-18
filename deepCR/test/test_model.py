import pytest

import numpy as np

from .. import model


def test_deepCR():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((299, 299))
    out = mdl.clean(in_im)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False)
    assert out.shape == in_im.shape


def test_seg():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((4000, 4000))
    out = mdl.clean(in_im, seg=256)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False, seg=256)
    assert out.shape == in_im.shape

def test_consistency():
    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((1000, 1000))
    out1 = mdl.clean(in_im, inpaint=False, binary=False)
    out2 = mdl.clean(in_im, seg=256, inpaint=False, binary=False)
    print(out1[0][:10])
    print(out2[0][:10])
    assert (out1 == out2).all()

if __name__ == '__main__':
    #test_deepCR()
    #test_seg()
    test_consistency()