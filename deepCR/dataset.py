import numpy as np
from torch.utils.data import Dataset

__all__ = ['dataset', 'DatasetSim']


class DatasetSim(Dataset):
    def __init__(self, image, cr, sky=None, aug_sky=(0, 0), aug_img=(1, 1), noise=False, saturation=1e5,
                 n_mask=1, norm=False, percentile_limit=50, part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param image: list of complete path to npy arrays, each containing one 2D image
        :param cr: list of complete path to npy arrays, each containing one 2D mask
        :param sky: np.ndarray [N,] or float, sky background level
        :param aug_sky: (float, float). If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param aug_img: (float, float). Multiply image (after sky augmentation) by a factor within that range
          with a log probability prior.
        :param noise: bool. Whether to add Poisson noise to image.
        :param saturation: float. maximum pixel value.
        :param n_mask: number of mask to stack for each image
        :param norm: bool. Whether to scale generated image by the standard deviation and median of
         part of the image below percentile_limit.
        :param percentile_limit: float. Upper percentile limit to calculate std and median, when norm=True
        :param part: either 'train' or 'val'. split by 0.8, 0.2
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """

        np.random.seed(seed)
        self.len_image = len(image)
        self.len_mask = len(cr)
        self.n_mask = n_mask
        self.noise = noise
        self.saturation = saturation

        assert f_val < 1 and f_val > 0
        f_train = 1 - f_val
        if part == 'train':
            s = np.s_[:int(self.len_image * f_train)]
            s_cr = np.s_[:int(self.len_mask * f_train)]
        elif part == 'val':
            s = np.s_[int(self.len_image * f_train):]
            s_cr = np.s_[:int(self.len_mask * f_train)]
        else:
            s = np.s_[0:]
            s_cr = np.s_[0:]

        if sky is None:
            sky = np.zeros(self.len_image)
        elif type(sky) != np.ndarray:
            sky = np.array([sky]*self.len_image)

        self.image = image[s]
        self.cr = cr[s_cr]
        self.sky = sky[s]
        self.aug_sky = aug_sky
        self.aug_img = (np.log10(aug_img[0]), np.log10(aug_img[1]))
        self.norm = norm
        self.percentile_limit = percentile_limit

        self.len_image = len(self.image)
        self.len_mask = len(self.cr)

    def get_cr(self):
        """
        Generate cosmic ray images from stacked dark frames.
        Sample self.n_mask number of cr masks and add them together.
        :return: (cr_img, cr_mask): sampled and added dark image containing only cr and accumulative cr mask
        """
        cr_id = np.random.randint(0, self.len_mask, self.n_mask)
        crs = []; masks = []
        for i in cr_id:
            arr = np.load(self.cr[i])
            crs.append(arr[0][None,:])
            masks.append(arr[1][None,:])
        masks = np.concatenate(masks).sum(axis=0) > 0
        crs = np.concatenate(crs).sum(axis=0)
        return crs, masks

    def get_image(self, i):
        data = np.load(self.image[i])
        if len(data.shape) == 3:
            return data[0], data[1]
        else:
            return data, np.zeros_like(data)

    def __len__(self):
        return self.len_image

    def __getitem__(self, i):
        cr, mask = self.get_cr()
        image, ignore = self.get_image(i)
        f_bkg_aug = self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])
        f_img_aug = self.aug_img[0] + np.random.rand() * (self.aug_img[1] - self.aug_img[0])
        f_img_aug = 10**f_img_aug
        bkg = f_bkg_aug * self.sky[i]
        img = (image + bkg) * f_img_aug + cr
        scale = img.copy()
        scale[scale < 1] = 1.
        scale = scale**0.5
        if self.noise:
            noise = np.random.normal(0, scale, image.shape)
        else:
            noise = np.zeros_like(image)
        img += noise
        img[img > self.saturation] = self.saturation

        if self.norm:
            limit = np.percentile(img, self.percentile_limit)
            clip = img[img < limit]
            scale = clip.std()
            median = np.percentile(clip, 50)
            img -= median
            img /= scale

        return img, mask, ignore


class dataset(Dataset):
    def __init__(self, image, mask, ignore=None, sky=None, aug_sky=[0, 0], part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param image: image with CR
        :param mask: CR mask
        :param ignore: loss mask, e.g., bad pixel, saturation, etc.
        :param sky: (np.ndarray) [N,] sky background level
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param part: either 'train' or 'val'. split by 0.8, 0.2
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """

        np.random.seed(seed)
        len = image.shape[0]
        assert f_val < 1 and f_val > 0
        f_train = 1 - f_val
        if sky is None:
            sky = np.zeros_like(image)
        if ignore is None:
            ignore = np.zeros_like(image)

        if part == 'train':
            s = np.s_[:max(1, int(len * f_train))]
        elif part == 'val':
            s = np.s_[min(len - 1, int(len * f_train)):]
        else:
            s = np.s_[0:]

        self.image = image[s]
        self.mask = mask[s]
        self.ignore = ignore[s]
        self.sky = sky[s]

        self.aug_sky = aug_sky

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, i):
        a = (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky[i]
        return self.image[i] + a, self.mask[i], self.ignore[i]
