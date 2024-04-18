from deepCR.unet_batchnorm import UNet2SigmoidBatchNorm, UNet2BatchNorm
from deepCR.unet import UNet2Sigmoid
from os import path

__all__ = ('mask_dict', 'inpaint_dict', 'default_model_path')

mask_dict = {'ACS-WFC': [UNet2SigmoidBatchNorm, (1, 1, 32), 1],
             'decam': [UNet2SigmoidBatchNorm, (1, 1, 32), 1],
             'example_model': [UNet2SigmoidBatchNorm, (1, 1, 32), 100],
             'WFC3-UVIS': [UNet2Sigmoid, (1, 1, 32), 100]
}

inpaint_dict = {'ACS-WFC-F606W-2-32': [UNet2BatchNorm, (2, 1, 32)]
}

default_model_path = path.join(path.dirname(__file__))
