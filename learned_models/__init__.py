from deepCR.unet import UNet2Sigmoid, UNet3, UNet2
from os import path

__all__ = ('mask_dict', 'inpaint_dict')

mask_dict = {'ACS-WFC-2-32': [UNet2Sigmoid, (1, 1, 32), 100],
             'ACS-WFC-2-4': [UNet2Sigmoid, (1, 1, 4), 100],
             'example_model': [UNet2Sigmoid, (1, 1, 32), 100]}

inpaint_dict = {'ACS-WFC-3-32': [UNet3, (2, 1, 32)],
                'ACS-WFC-2-32': [UNet2, (2, 1, 32)]}

default_model_path = path.join(path.dirname(__file__))
