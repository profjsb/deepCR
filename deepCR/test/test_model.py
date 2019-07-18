import time

import numpy as np
import pytest

from .. import model


def test_deepCR_serial():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((299, 299))
    out = mdl.clean(in_im)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    out = mdl.clean(in_im, inpaint=False)
    assert out.shape == in_im.shape

def test_deepCR_parallel():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((512, 512))

    out = mdl.clean(in_im, parallel=True, seg=256, n_jobs=-1)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    # Is the serial runtime slower than the parallel runtime on a big image?
    in_im = np.ones((3096, 2000))
    t0 = time.time()
    out = mdl.clean(in_im, inpaint=False, parallel=True, seg=256)
    par_runtime = time.time() - t0
    assert out.shape == in_im.shape

    t0 = time.time()
    out = mdl.clean(in_im, inpaint=False, parallel=False, seg=256)
    ser_runtime = time.time() - t0
    # assert False, f"par={par_runtime}, ser={ser_runtime}"
    assert par_runtime < ser_runtime


def test_seg():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((3999, 3999))
    out = mdl.clean(in_im, seg=256)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False, seg=512)
    assert out.shape == in_im.shape


#if __name__ == '__main__':
#    test_deepCR_serial()
#    test_deepCR_parallel()
#    test_seg()

