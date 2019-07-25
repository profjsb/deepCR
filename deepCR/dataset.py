from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
    def __init__(self, image, mask, use, sky=None, part='train', aug_sky=[0,0]):

        np.random.seed(1)
        len = image.shape[0]
        if sky is None:
            sky = np.zeros_like(image)
        if part == 'train':
            self.image = image[:int(len * 0.8)]
            self.mask = mask[:int(len * 0.8)]
            self.use = use[:int(len * 0.8)]
            self.sky = sky[:int(len * 0.8)]
        else:
            self.image = image[:int(len * 0.2)]
            self.mask = mask[:int(len * 0.2)]
            self.use = use[:int(len * 0.2)]
            self.sky = sky[:int(len * 0.2)]

        self.aug_sky = aug_sky

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, i):
        a = (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0]))* self.sky[i]
        return self.image[i] + a, self.mask[i], self.use[i]