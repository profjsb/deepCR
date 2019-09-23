import os
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
    trainer.train_phase1(1)
    assert trainer.epoch_mask == 3


def test_train_sim():
    cwd = os.getcwd()+'/'

    os.mkdir('temp')
    os.mkdir('temp/image')
    os.mkdir('temp/dark')

    var = np.zeros((2, 64, 64))
    for i in range(6):
        np.save(cwd + 'temp/image/%d.npy'%i, var)
        np.save(cwd + 'temp/dark/%d.npy'%i, var)
    image_list = [cwd + 'temp/image/' + f for f in os.listdir(cwd + 'temp/image')]
    dark_list = [cwd + 'temp/dark/' + f for f in os.listdir(cwd + 'temp/dark')]
    trainer = train(image=image_list, mask=dark_list, sky=100, aug_sky=[-0.9, 10], aug_img=[0.5, 5],
                    n_mask=2, epoch=2, verbose=False)
    trainer.train()
    filename = trainer.save()
    trainer.load(filename)
    trainer.train_phase1(1)
    assert trainer.epoch_mask == 3

    for root, dirs, files in os.walk(cwd + 'temp', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir('temp')


if __name__ == '__main__':
    test_train()
    test_train_sim()
