import os
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
    # Make sure we have a lot of cores
    # otherwise this can fail on Travis b/c we only get 1-2 at test time.
    if os.cpu_count() > 2:
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
    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((4000, 4000))
    out = mdl.clean(in_im, seg=256)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False, seg=256)
    assert out.shape == in_im.shape

"""
def test_consistency():
    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((1000, 1000))
    out1 = mdl.clean(in_im, inpaint=False, binary=False)
    out2 = mdl.clean(in_im, seg=256, inpaint=False, binary=False)
    print(out1[0][:10], type(out1[0][0]))
    print(out2[0][:10], type(out2[0][0]))
    assert (out1 == out2).all()
"""
#if __name__ == '__main__':
#    test_consistency()
