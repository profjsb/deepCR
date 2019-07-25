import os
import time

import numpy as np
import pytest

from .. import model

def test_deepCR_serial():

    mdl = model.deepCR(mask='ACS-WFC-2-32', device='CPU')
    in_im = np.ones((299, 299))
    out = mdl.clean(in_im)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    out = mdl.clean(in_im, inpaint=False)
    assert out.shape == in_im.shape

def test_deepCR_parallel():

    mdl = model.deepCR(mask='ACS-WFC-2-32', device='CPU')
    in_im = np.ones((299, 299))
    out = mdl.clean(in_im, parallel=True)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)

    # Is the serial runtime slower than the parallel runtime on a big image?
    # Make sure we have a lot of cores
    # otherwise this can fail on Travis b/c we only get 1-2 at test time.
    if os.cpu_count() > 2:
        in_im = np.ones((3096, 2000))
        t0 = time.time()
        out = mdl.clean(in_im, inpaint=False, parallel=True)
        par_runtime = time.time() - t0
        assert out.shape == in_im.shape

        t0 = time.time()
        out = mdl.clean(in_im, inpaint=False, parallel=False)
        ser_runtime = time.time() - t0
        assert par_runtime < ser_runtime

def test_seg():
    mdl = model.deepCR(mask='ACS-WFC-2-32', inpaint='ACS-WFC-2-32', device='CPU')
    in_im = np.ones((500, 1000))
    out = mdl.clean(in_im, segment=True)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
    out = mdl.clean(in_im, inpaint=False, segment=True)
    assert out.shape == in_im.shape

if __name__ == '__main__':
    test_seg()
    test_deepCR_parallel()
    test_deepCR_serial()
