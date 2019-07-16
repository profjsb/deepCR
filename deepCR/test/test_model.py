import pytest

import numpy as np

from .. import model

def test_deepCR():

    mdl = model.deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
    in_im = np.ones((256, 256))
    out = mdl.clean(in_im)
    assert (out[0].shape, out[1].shape) == (in_im.shape, in_im.shape)
