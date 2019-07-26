import os

import numpy as np
import pytest

from .. import training

def test_train():

    inputs = np.ones((6,64,64))
    sky = np.ones(6)
    trainer = training.train(image=inputs, mask=inputs, sky=sky, aug_sky=[-0.9, 10], epoch=2)
    trainer.train()

if __name__ == '__main__':
    test_train()