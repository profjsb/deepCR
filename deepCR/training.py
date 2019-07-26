""" module for training new deepCR-mask models
"""

import torch
from tqdm import tqdm_notebook as tqdm
from deepCR.unet import WrappedModel, UNet2Sigmoid
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import datetime
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from deepCR.util import maskMetric
from deepCR.dataset import dataset

__all__ = ('train')

class train():

    def __init__(self, image, mask, use=None, sky=None, name='model', hidden=32, gpu=False, epoch=50, batch_size=16, lr=0.005, aug_sky=[0, 0], save_after=1e3, every=10, directory='./'):
        """ This is the class for training deepCR-mask.
        :param image: np.ndarray (N*W*W) training data: image array with CR.
        :param mask: np.ndarray (N*W*W) training data: CR mask array
        :param use: training data: Mask for taking loss. e.g., bad pixel, saturation, etc.
        :param name: model name. model saved to name_epoch.pth
        :param hidden: number of channels for the first convolution layer. default: 50
        :param gpu: True if use GPU for training
        :param epoch: Number of epochs to train. default: 50
        :param batch_size: training batch size. default: 16
        :param lr: learning rate. default: 0.005
        :param aug_sky: [float, float]. Use random sky background from aug_sky[0] * sky to aug_sky[1] * sky.
        :param save_after: epoch after which trainer automatically saves model state with lowest validation loss
        :param every: for every epoch, visualize mask prediction for 1st image in validation set.
        """
        if sky is None and aug_sky != [0, 0]:
            raise AttributeError('Var (sky) is required for sky background augmentation!')
        if use is None:
            use = np.zeros_like(image)
        assert image.shape == mask.shape == use.shape
        assert image.shape[1] == image.shape[2]
        data_train = dataset(image, mask, use, sky, part='train', aug_sky=aug_sky)
        data_val = dataset(image, mask, use, sky, part='val', aug_sky=aug_sky)
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=1)
        self.shape = image.shape[1]
        self.name = name

        if gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            self.network = nn.DataParallel(UNet2Sigmoid(1,1,hidden))
            self.network.type(self.dtype)
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            self.network = WrappedModel(UNet2Sigmoid(1,1,hidden))
            self.network.type(self.dtype)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=4, cooldown=2,
                                                          min_lr=lr * 1e-4, verbose=True, threshold=0.01)

        self.BCELoss = nn.BCELoss()
        self.lossMask_val = []
        self.epoch_mask = 0
        self.save_after = save_after
        self.n_epochs = epoch
        self.every = every
        self.directory = directory

    def set_input(self, img0, mask, use):
        """
        :param img0: input image
        :param mask: CR mask
        :param use: loss mask
        :return: None
        """
        self.img0 = Variable(img0.type(self.dtype)).view(-1, 1, self.shape, self.shape)
        self.mask = Variable(mask.type(self.dtype)).view(-1, 1, self.shape, self.shape)
        self.use = Variable(use.type(self.dtype)).view(-1, 1, self.shape, self.shape)

    def validate_mask(self):
        """
        :return: validation loss. print TPR and FPR at threshold = 0.5.
        """
        lmask = 0; count = 0
        metric = np.zeros(4)
        for i, dat in enumerate(self.ValLoader):
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            loss = self.backward_network()
            lmask += float(loss.detach()) * n
            metric += maskMetric(self.pdt_mask.reshape(-1, self.shape, self.shape).detach().cpu().numpy() > 0.5, dat[1].numpy())
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        print('TPR=%.3f   FPR=%.3f' % (TPR * 100, FPR * 100))
        return (lmask)

    def train(self):
        """ call this function to start training network
        :return: None
        """
        self.network.train()
        for epoch in tqdm(range(int(self.n_epochs * 0.4 + 0.5))):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every==0:
                plt.figure(figsize=(10, 30))
                plt.subplot(131)
                plt.imshow(np.log(self.img0[0, 0].detach().cpu().numpy()), cmap='gray')
                plt.subplot(132)
                plt.imshow(self.pdt_mask[0, 0].detach().cpu().numpy()>0.5, cmap='gray')
                plt.subplot(133)
                plt.imshow(self.mask[0, 0].detach().cpu().numpy(), cmap='gray')
                plt.show()

            print('epoch = %d' % (self.epoch_mask))
            valLossMask = self.validate_mask()
            self.lossMask_val.append(valLossMask)
            if (np.array(self.lossMask_val)[-1] == np.array(
                    self.lossMask_val).min() and self.epoch_mask > self.save_after):
                self.save_mask()
            self.lr_scheduler.step(self.lossMask_val[-1])
            print('loss = %.4f' % (self.lossMask_val[-1]))
            self.n_epochs -= 1

        print('Network set to evaluation mode; BN parameter frozen')
        self.network.eval()
        for epoch in tqdm(range(self.n_epochs - int(self.n_epochs * 0.4 + 0.5))):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every==0:
                plt.figure(figsize=(10, 30))
                plt.subplot(131)
                plt.imshow(np.log(self.img0[0, 0].detach().cpu().numpy()), cmap='gray')
                plt.subplot(132)
                plt.imshow(self.pdt_mask[0, 0].detach().cpu().numpy()>0.5, cmap='gray')
                plt.subplot(133)
                plt.imshow(self.mask[0, 0].detach().cpu().numpy(), cmap='gray')
                plt.show()

            print('epoch = %d' % (self.epoch_mask))
            valLossMask = self.validate_mask()
            self.lossMask_val.append(valLossMask)
            if (np.array(self.lossMask_val)[-1] == np.array(
                    self.lossMask_val).min() and self.epoch_mask > self.save_after):
                self.save_mask()
            self.lr_scheduler.step(self.lossMask_val[-1])
            print('loss = %.4f' % (self.lossMask_val[-1]))
            self.n_epochs -= 1

    def optimize_network(self, dat):
        self.set_input(*dat)
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()

    def backward_network(self):
        loss = self.BCELoss(self.pdt_mask * (1 - self.use), self.mask * (1 - self.use))
        return (loss)

    def plot_loss(self):
        """ plot validation loss vs. epoch
        :return: None
        """
        plt.figure(figsize=(10,5))
        plt.plot(range(self.epoch_mask), self.lossMask_val)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Validation loss')
        plt.show()

    def save_mask(self):
        """ save trained network parameters to date_model_name_epoch*.pth
        :return: None
        """
        time = datetime.datetime.now()
        time = str(time)[:10]
        filename = '%s_%s_epoch%d' % (time, self.name, self.epoch_mask)
        print(filename)
        torch.save(self.network.state_dict(), self.directory + filename + '.pth')
        lossTrain = np.array(self.lossMask_train)
        lossVal = np.array(self.lossMask_val)
        np.save(self.directory + filename + 'loss.npy', lossVal)

    def load_mask(self, filename):
        """ Continue training from a previous model state saved to filename
        :param filename: (str) filename (without ".pth") to load model state
        :return: None
        """
        self.network.load_state_dict(torch.load(self.directory + filename + '.pth'))
        self.lossMask_val = list(np.load(self.directory + filename + 'loss.npy'))
        loc = filename.find('epoch') + 5
        self.epoch_mask = int(filename[loc:])