"""
deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Apply a learned DL model to a 2d numpy array to remove
cosmic rays.

Code which accompanies the paper: Zhang & Bloom (2019)

See https://github.com/profjsb/deepCR

"""
from . import model, evaluate, training

__all__ = [model.__all__, evaluate.__all__, training.__all__]

__version__ = '0.1.4'
