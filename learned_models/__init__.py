from deepCR.unet import UNet2Sigmoid, UNet3, UNet2
from os import path

__all__ = ('mask_dict', 'inpaint_dict')

mask_dict = {'ACS-WFC-F606W-2-32': [UNet2Sigmoid, (1, 1, 32)],
             'ACS-WFC-F606W-2-4': [UNet2Sigmoid, (1, 1, 4)],
             'ACS-WFC-F606W-2-32-old': [UNet2Sigmoid, (1, 1, 32)],
             'ACS-WFC-F606W-2-4-old': [UNet2Sigmoid, (1, 1, 4)]}

inpaint_dict = {'ACS-WFC-F606W-3-32': [UNet3, (2, 1, 32)],
                'ACS-WFC-F606W-2-32': [UNet2, (2, 1, 32)],
                'ACS-WFC-F606W-3-32-old': [UNet3, (2, 1, 32)],
                'ACS-WFC-F606W-2-32-old': [UNet2, (2, 1, 32)]}

default_model_path = path.join(path.dirname(__file__))
