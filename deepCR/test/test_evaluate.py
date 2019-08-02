import numpy as np
import pytest

import deepCR.evaluate as evaluate
from deepCR.model import deepCR


def test_eval():
    mdl = deepCR()
    var = np.zeros((10,24,24))
    tpr, fpr = evaluate.roc(mdl, image=var, mask=var, thresholds=np.linspace(0,1,10))
    assert tpr.shape == (10,)


if __name__ == '__main__':
    test_eval()
