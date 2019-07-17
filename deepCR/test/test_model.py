import pytest

import numpy as np

from .. import model


def test_deepCR():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((256, 256))
    out = mdl.clean(in_im)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False)
    assert out.shape == in_im.shape


def test_seg():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((1024, 1024))
    out = mdl.clean(in_im, seg=256)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False, seg=512)
    assert out.shape == in_im.shape


#if __name__ == '__main__':
#    test_deepCR()
#    test_seg()