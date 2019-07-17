from os import path

import numpy as np
import torch
import torch.nn as nn
from torch import from_numpy

from deepCR.unet import WrappedModel
from deepCR.util import medmask
from learned_models import mask_dict, inpaint_dict, default_model_path

__all__ = ('deepCR', 'mask_dict', 'inpaint_dict', 'default_model_path')


class deepCR():

    def __init__(self, mask=None, inpaint='medmask', device='CPU',
                 model_dir=default_model_path):

        """model class instantiation for deepCR. Here is the
    declaration of the learned mask and inpainting models
    to be used on images

    Parameters
    ----------
    mask : str
        Name of the mask to use. This is one of the keys in
        `mask_dict`
    inpaint : str
        Name of the inpainting model to use. This is one of the keys in
        `inpaint_dict`. It can also be `medmask` which will then
        use a simple 5x5 median mask for inpainting
    device : str
        One of 'CPU' or 'GPU'
    model_dir : str
        The location of the model directory with the mask/ and inpaint/
        subdirectories. This defaults to where the pre-shipped
        models live (in `learned_models/`)

    Returns
    -------
    None
        """
        if(device == 'GPU'):
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            wrapper = nn.DataParallel
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            wrapper = WrappedModel

        if mask is not None:
            self.maskNet = wrapper(mask_dict[mask][0](*mask_dict[mask][1]))
            self.maskNet.type(self.dtype)
            if device != 'GPU':
                self.maskNet.load_state_dict(torch.load(model_dir + '/mask/' + mask + '.pth',
                                                        map_location='cpu'))
            else:
                self.maskNet.load_state_dict(torch.load(model_dir + '/mask/' + mask + '.pth'))

            self.maskNet.eval()
            for p in self.maskNet.parameters():
                p.required_grad = False

        if inpaint == 'medmask':
            self.inpaintNet = None
        else:
            self.inpaintNet = wrapper(inpaint_dict[inpaint][0](*inpaint_dict[inpaint][1])).type(self.dtype)
            if device != 'GPU':
                self.inpaintNet.load_state_dict(torch.load(model_dir+'/inpaint/' + inpaint+'.pth',
                                                           map_location='cpu'))
            else:
                self.inpaintNet.load_state_dict(torch.load(model_dir+'/inpaint/' + inpaint+'.pth'))
            self.inpaintNet.eval()
            for p in self.inpaintNet.parameters():
                p.required_grad = False

    def clean(self, img0, threshold=0.5, inpaint=True, binary=True):

        """
            given input image
            return cosmic ray mask and (optionally) clean image
            mask could be binary or probablistic
        :param img0 (np.ndarray): 2D input image
        :param threshold: for creating binary mask from probablistic mask
        :param inpaint: return clean image only if True
        :param binary: return binary mask if True. probabilistic mask otherwise.
        :return: mask or binary mask; or None if internal call
        """
        shape = img0.shape[-2:]
        img0 = from_numpy(img0).type(self.dtype).view(1, -1, shape[0], shape[1])
        mask = self.maskNet(img0)

        if not binary:
            return mask.detach().cpu().view(shape[0], shape[1]).numpy()

        binary_mask = (mask > threshold).type(self.dtype)

        if inpaint:
            if self.inpaintNet is not None:
                cat = torch.cat((img0 * (1 - binary_mask), binary_mask), dim=1)
                img1 = self.inpaintNet(cat)
                img1 = img1.detach()
                inpainted = img1 * binary_mask + img0 * (1 - binary_mask)
                binary_mask = binary_mask.detach().cpu().view(shape[0], shape[1]).numpy()
                inpainted = inpainted.detach().cpu().view(shape[0], shape[1]).numpy()
            else:
                binary_mask = binary_mask.detach().cpu().view(shape[0], shape[1]).numpy()
                img0 = img0.detach().cpu().view(shape[0], shape[1]).numpy()
                img1 = medmask(img0, binary_mask)
                inpainted = img1 * binary_mask + img0 * (1 - binary_mask)

            if binary:
                return (binary_mask, inpainted)
            else:
                mask = mask.detach().cpu().view(shape[0], shape[1]).numpy()
                return (mask, inpainted)

        else:
            if binary:
                return binary_mask
            else:
                mask = mask.detach().cpu().view(shape[0], shape[1]).numpy()
                return mask

    def inpaint(self, img0, mask):

        """
            inpaint parts of an image given an inpaint mask
        :param img0 (np.ndarray): 2D input image
        :param mask (np.ndarray): 2D input mask
        :return: inpainted image
        """
        shape = img0.shape[-2:]
        if self.inpaintNet is not None:
            img0 = from_numpy(img0).type(self.dtype). \
                              view(1, -1, shape[0], shape[1])
            mask = from_numpy(mask).type(self.dtype). \
                   view(1, -1, shape[0], shape[1])
            cat = torch.cat((img0 * (1 - mask), mask), dim=1)
            img1 = self.inpaintNet(cat)
            img1 = img1.detach()
            inpainted = img1 * mask + img0 * (1 - mask)
            inpainted = inpainted.detach().cpu(). \
                   view(shape[0], shape[1]).numpy()
        else:
            img1 = medmask(img0, mask)
            inpainted = img1 * mask + img0 * (1 - mask)

        return inpainted
