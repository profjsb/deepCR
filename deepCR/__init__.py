"""
deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Apply a learned DL model to a 2d numpy array to remove
cosmic rays.

Code which accompanies the paper: Zhang & Bloom (2019)

See https://github.com/profjsb/deepCR

"""
from deepCR.model import deepCR
from deepCR.training import train
from deepCR.evaluate import roc

__all__ = [deepCR, train, roc]

__version__ = '0.1.4'
