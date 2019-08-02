import os

import numpy as np
import pytest

from deepCR.training import train


def test_train():
    inputs = np.zeros((6, 64, 64))
    sky = np.ones(6)
    trainer = train(image=inputs, mask=inputs, sky=sky, aug_sky=[-0.9, 10], epoch=2, verbose=False)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    test_train()
