import pytest

import numpy as np

from .. import util


def test_medmask():

    in_im = np.random.random((256, 256))
    mask = np.random.randint(0,2, size=(256, 256))

    masked = util.medmask(in_im, mask)

    ok_inds = np.where(mask == 0)

    # make sure that the good pixels were not masked
    assert np.all(in_im[ok_inds] == masked[ok_inds])
