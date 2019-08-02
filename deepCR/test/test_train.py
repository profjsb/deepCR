import numpy as np
import pytest

from deepCR.training import train


def test_train():
    inputs = np.zeros((6, 64, 64))
    sky = np.ones(6)
    trainer = train(image=inputs, mask=inputs, sky=sky, aug_sky=[-0.9, 10], verbose=False, epoch=2, save_after=10)
    trainer.train()
    filename = trainer.save()
    trainer.load(filename)
    trainer.train_continue(1)
    assert trainer.epoch_mask == 3


if __name__ == '__main__':
    test_train()
