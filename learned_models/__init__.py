from deepCR.unet import UNet2Sigmoid, UNet3, UNet2
from os import path

__all__ = ('mask_dict', 'inpaint_dict', 'default_model_path')

mask_dict = {'ACS-WFC-F435W-2-32': [UNet2Sigmoid, (1, 1, 32), 100],
             'ACS-WFC-F606W-2-32': [UNet2Sigmoid, (1, 1, 32), 100],
             'ACS-WFC-F814W-2-32': [UNet2Sigmoid, (1, 1, 32), 100],
             'ACS-WFC-2-32': [UNet2Sigmoid, (1, 1, 32), 100],
             'ACS-WFC': [UNet2Sigmoid, (1, 1, 32), 1],
             'ACS-WFC-F606W-2-4': [UNet2Sigmoid, (1, 1, 4), 100],
             'decam': [UNet2Sigmoid, (1, 1, 32), 1],
             'example_model': [UNet2Sigmoid, (1, 1, 32), 100]}

inpaint_dict = {'ACS-WFC-F606W-3-32': [UNet3, (2, 1, 32)],
                'ACS-WFC-F606W-2-32': [UNet2, (2, 1, 32)]}

default_model_path = path.join(path.dirname(__file__))
