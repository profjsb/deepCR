from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
    def __init__(self, image, mask, ignore=None, sky=None, part='train', aug_sky=[0, 0]):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param image: image with CR
        :param mask: CR mask
        :param ignore: loss mask, e.g., bad pixel, saturation, etc.
        :param sky: (np.ndarray) [N,] sky background level
        :param part: either 'train' or 'val'. split by 0.8, 0.2
        :param aug_sky: [negative number, positive number]. Add sky background by aug_sky[0] * sky to aug_sky[1] * sky.
        """
        np.random.seed(1)
        len = image.shape[0]
        if sky is None:
            sky = np.zeros_like(image)
        if ignore is None:
            ignore = np.zeros_like(image)
        if part == 'train':
            self.image = image[:int(len * 0.8)]
            self.mask = mask[:int(len * 0.8)]
            self.ignore = ignore[:int(len * 0.8)]
            self.sky = sky[:int(len * 0.8)]
        elif part == 'val':
            self.image = image[:int(len * 0.2)]
            self.mask = mask[:int(len * 0.2)]
            self.ignore = ignore[:int(len * 0.2)]
            self.sky = sky[:int(len * 0.2)]
        else:
            self.image = image
            self.mask = mask
            self.ignore = ignore
            self.sky = sky

        self.aug_sky = aug_sky

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, i):
        a = (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky[i]
        return self.image[i] + a, self.mask[i], self.ignore[i]
