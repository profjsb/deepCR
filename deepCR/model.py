"""main module to instantiate deepCR models and use them
"""
from os import path, mkdir
import math
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch import from_numpy
from joblib import Parallel, delayed
from joblib import dump, load
from joblib import wrap_non_picklable_objects

from tqdm import tqdm


from deepCR.unet import WrappedModel
from deepCR.util import medmask
from learned_models import mask_dict, inpaint_dict, default_model_path

__all__ = ('deepCR', 'mask_dict', 'inpaint_dict', 'default_model_path')


class deepCR():

    def __init__(self, mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-3-32', device='CPU',
                 model_dir=default_model_path):

        """
            Instantiation of deepCR with specified model configurations

        Parameters
        ----------
        mask : str
            Name of deepCR-mask model to use.
        inpaint : str
            Name of the inpainting model to use. It can also be `medmask` which will then
            use a simple 5x5 median mask sampling for inpainting
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
        ifdevice == 'GPU':
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

    def clean(self, img0, threshold=0.5, inpaint=True, binary=True, segment=False,
              patch=256, parallel=False, n_jobs=-1):
        """
            Identify cosmic rays in an input image, and (optionally) inpaint with the predicted cosmic ray mask
        :param img0: (np.ndarray) 2D input image conforming to model requirements. For HST ACS/WFC, must be from _flc.fits and in units of electrons in native resolution.
        :param threshold: (float; [0, 1]) applied to probabilistic mask to generate binary mask
        :param inpaint: (bool) return clean, inpainted image only if True
        :param binary: return binary CR mask if True. probabilistic mask if False
        :param segment: (bool) if True, segment input image into chunks of patch * patch before performing CR rejection. Used for memory control.
        :param patch: (int) Use 256 unless otherwise required. if segment==True, segment image into chunks of patch * patch.
        :param parallel: (bool) run in parallel if True and segment==True
        :param n_jobs: (int) number of jobs to run in parallel, passed to `joblib.` Beware of memory overflow for larger n_jobs.
        :return: CR mask and (optionally) clean inpainted image
        """

        # data pre-processing
        img0 = img0.astype(np.float32) / 100

        if not segment:
            return self.clean_(img0, threshold=threshold,
                               inpaint=inpaint, binary=binary)
        else:
            if not parallel:
                return self.clean_large(img0, threshold=threshold,
                               inpaint=inpaint, binary=binary, patch=patch)
            else:
                return self.clean_large_parallel(img0, threshold=threshold,
                               inpaint=inpaint, binary=binary, patch=patch,
                               n_jobs=n_jobs)

    def clean_(self, img0, threshold=0.5, inpaint=True, binary=True):

        """
            given input image
            return cosmic ray mask and (optionally) clean image
            mask could be binary or probabilistic
        :param img0: (np.ndarray) 2D input image
        :param threshold: for creating binary mask from probabilistic mask
        :param inpaint: return clean image only if True
        :param binary: return binary mask if True. probabilistic mask otherwise.
        :return: CR mask and (optionally) clean inpainted image
        """

        shape = img0.shape
        pad_x = 4 - shape[0] % 4
        pad_y = 4 - shape[1] % 4
        if pad_x == 4:
            pad_x = 0
        if pad_y == 4:
            pad_y = 0
        img0 = np.pad(img0, ((pad_x, 0), (pad_y, 0)), mode='constant')

        shape = img0.shape[-2:]
        img0 = from_numpy(img0).type(self.dtype).view(1, -1, shape[0], shape[1])
        mask = self.maskNet(img0)

        if not binary:
            return mask.detach().cpu().view(shape[0], shape[1]).numpy()[pad_x:, pad_y:]

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
                return binary_mask[pad_x:, pad_y:], inpainted[pad_x:, pad_y:] * 100
            else:
                mask = mask.detach().cpu().view(shape[0], shape[1]).numpy()
                return mask[pad_x:, pad_y:], inpainted[pad_x:, pad_y:] * 100

        else:
            if binary:
                binary_mask = binary_mask.detach().cpu().view(shape[0], shape[1]).numpy()
                return binary_mask[pad_x:, pad_y:]
            else:
                mask = mask.detach().cpu().view(shape[0], shape[1]).numpy()
                return mask[pad_x:, pad_y:]

    def clean_large_parallel(self, img0, threshold=0.5, inpaint=True, binary=True,
                    patch=256, n_jobs=-1):
        """
            given input image
            return cosmic ray mask and (optionally) clean image
            mask could be binary or probabilistic
        :param img0: (np.ndarray) 2D input image
        :param threshold: for creating binary mask from probabilistic mask
        :param inpaint: return clean image only if True
        :param binary: return binary mask if True. probabilistic mask otherwise.
        :param patch: (int) Use 256 unless otherwise required. patch size to run deepCR on.
        :param n_jobs: (int) number of jobs to run in parallel, passed to `joblib.` Beware of memory overflow for larger n_jobs.
        :return: CR mask and (optionally) clean inpainted image
        """
        folder = './joblib_memmap'
        try:
            mkdir(folder)
        except FileExistsError:
            pass

        im_shape = img0.shape
        img0_dtype = img0.dtype
        hh = int(math.ceil(im_shape[0]/patch))
        ww = int(math.ceil(im_shape[1]/patch))

        img0_filename_memmap = path.join(folder, 'img0_memmap')
        dump(img0, img0_filename_memmap)
        img0 = load(img0_filename_memmap, mmap_mode='r')

        if inpaint:
            img1_filename_memmap = path.join(folder, 'img1_memmap')
            img1 = np.memmap(img1_filename_memmap, dtype=img0.dtype,
                            shape=im_shape, mode='w+')
        else:
            img1 = None

        mask_filename_memmap = path.join(folder, 'mask_memmap')
        mask = np.memmap(mask_filename_memmap, dtype=np.int8 if binary else img0_dtype,
                           shape=im_shape, mode='w+')

        @wrap_non_picklable_objects
        def fill_values(i, j, img0, img1, mask, patch, inpaint, threshold, binary):
            img = img0[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])]
            if inpaint:
                mask_, clean_ = self.clean_(img, threshold=threshold, inpaint=True, binary=binary)
                mask[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])] = mask_
                img1[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])] = clean_
            else:
                mask_ = self.clean_(img, threshold=threshold, inpaint=False, binary=binary)
                mask[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])] = mask_

        results = Parallel(n_jobs=n_jobs, verbose=0)\
                   (delayed(fill_values)(i, j, img0, img1, mask, patch, inpaint, threshold, binary)
                    for i in range(hh) for j in range(ww))

        mask = np.array(mask)
        if inpaint:
            img1 = np.array(img1)
        try:
            shutil.rmtree(folder)
        except:
            print('Could not clean-up automatically.')

        if inpaint:
            return mask, img1
        else:
            return mask

    def clean_large(self, img0, threshold=0.5, inpaint=True, binary=True,
                    patch=256):
        """
            given input image
            return cosmic ray mask and (optionally) clean image
            mask could be binary or probabilistic
        :param img0: (np.ndarray) 2D input image
        :param threshold: for creating binary mask from probabilistic mask
        :param inpaint: return clean image only if True
        :param binary: return binary mask if True. probabilistic mask otherwise.
        :return: mask or binary mask; or None if internal call
        """
        im_shape = img0.shape
        hh = int(math.ceil(im_shape[0]/patch))
        ww = int(math.ceil(im_shape[1]/patch))

        img1 = np.zeros((im_shape[0], im_shape[1]))
        mask = np.zeros((im_shape[0], im_shape[1]))

        if inpaint:
            for i in range(hh):
                for j in range(ww):
                    img = img0[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])]
                    mask_, clean_ = self.clean_(img, threshold=threshold, inpaint=True, binary=binary)
                    mask[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])] = mask_
                    img1[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])] = clean_
            return mask, img1

        else:
            for i in range(hh):
                for j in range(ww):
                    img = img0[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])]
                    mask_ = self.clean_(img, threshold=threshold, inpaint=False, binary=binary)
                    mask[i * patch: min((i + 1) * patch, im_shape[0]), j * patch: min((j + 1) * patch, im_shape[1])] = mask_
            return mask

    def inpaint(self, img0, mask):

        """
            inpaint img0 under mask
        :param img0: (np.ndarray) input image
        :param mask: (np.ndarray) inpainting mask
        :return: inpainted clean image
        """
        img0 = img0.astype(np.float32) / 100
        mask = mask.astype(np.float32)
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
        return inpainted * 100

