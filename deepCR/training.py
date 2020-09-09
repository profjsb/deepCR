""" module for training new deepCR-mask models
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from tqdm import tqdm_notebook as tqdm_notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from deepCR.util import maskMetric
from deepCR.dataset import dataset, DatasetSim, PairedDatasetImagePath
from deepCR.unet import WrappedModel, UNet2Sigmoid

__all__ = 'train'


class VoidLRScheduler:
    def __init__(self):
        pass

    def _reset(self):
        pass

    def step(self):
        pass


class train:

    def __init__(self, image, mask=None, ignore=None, sky=None, mode='pair', aug_sky=(0, 0), aug_img=(1, 1), noise=False, saturation=1e5,
                 n_mask_train=1, n_mask_val=1, norm=False, percentile_limit=50, name='model', hidden=32, epoch=50,
                 epoch_phase0=None, batch_size=16, lr=0.005, auto_lr_decay=True, lr_decay_patience=4,
                 lr_decay_factor=0.1, save_after=1e5, plot_every=10, verbose=True, use_tqdm=False,
                 use_tqdm_notebook=False, directory='./'):

        """ This is the class for training deepCR-mask.
        :param image: np.ndarray (N*W*W) of CR affected images or list of npy arrays of shape (2or3,W,W), where
        the 1st dim is CR affect image, 2nd dim is CR mask, (optional) 3rd dimension is ignore mask.
        See documentation: training tutorial.
        :param mask: np.ndarray (N*W*W) training data: CR mask array.
        :param ignore: training data: Mask for taking loss. e.g., bad pixel, saturation, etc.
        :param sky: np.ndarray (N,) (optional) sky background. When param: iamge is list to npy files, provide sky level
        as sky.npy in each image subdirectory (see documentation: training tutorial).
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param n_mask: number of dark images to sample to create one cosmic ray image and mask
        :param name: model name. model saved to name_epoch.pth
        :param hidden: number of channels for the first convolution layer. default: 50
        :param gpu: True if use GPU for training
        :param epoch: Number of epochs to train. default: 50
        :param batch_size: training batch size. default: 16
        :param lr: learning rate. default: 0.005
        :param auto_lr_decay: reduce learning rate by "lr_decay_factor" after validation loss do not decrease for
          "lr_decay_patience" + 1 epochs.
        :param lr_decay_patience: reduce learning rate by lr_decay_factor after validation loss do not decrease for
          "lr_decay_patience" + 1 epochs.
        :param lr_decay_factor: multiplicative factor by which to reduce learning rate.
        :param save_after: epoch after which trainer automatically saves model state with lowest validation loss
        :param plot_every: for every "plot_every" epoch, plot mask prediction and ground truth for 1st image in
          validation set.
        :param verbose: print validation loss and detection rates for every epoch.
        :param use_tqdm: whether to show tqdm progress bar.
        :param use_tqdm_notebook: whether to use jupyter notebook version of tqdm. Overwrites tqdm_default.
        :param directory: directory relative to current path to save trained model.
        """
        if torch.cuda.is_available():
            gpu = True
        else:
            gpu = False
            print('No GPU detected on this device! Training on CPU.')

        if sky is None and aug_sky != (0, 0):
            raise AttributeError('Var (sky) is required for sky background augmentation!')
        if ignore is None:
            ignore = np.zeros_like(image)
        if mode == 'pair':
            if type(image[0]) == str or type(image[0]) == np.str_:
                data_train = PairedDatasetImagePath(image, aug_sky[0], aug_sky[1], part='train')
                data_val = PairedDatasetImagePath(image, aug_sky[0], aug_sky[1], part='val')
            else:
                data_train = dataset(image, mask, ignore, sky, part='train', aug_sky=aug_sky)
                data_val = dataset(image, mask, ignore, sky, part='val', aug_sky=aug_sky)
        elif mode == 'simulate':
            data_train = DatasetSim(image, mask, sky, aug_sky=aug_sky, aug_img=aug_img, saturation=saturation,
                                    norm=norm, percentile_limit=percentile_limit, part='train', noise=noise, n_mask=n_mask_train)
            data_val = DatasetSim(image, mask, sky, aug_sky=aug_sky, aug_img=aug_img, saturation=saturation,
                                  norm=norm, percentile_limit=percentile_limit, part='val', noise=noise, n_mask=n_mask_val)
        else:
            raise TypeError('Mode must be one of pair or simulate!')

        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=8)
        self.shape = data_train[0][0].shape[1]
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
        if auto_lr_decay:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
                                                  cooldown=2, verbose=True, threshold=0.005)
        else:
            self.lr_scheduler = VoidLRScheduler()
        self.lr = lr
        self.BCELoss = nn.BCELoss()
        self.validation_loss = []
        self.epoch_mask = 0
        self.save_after = save_after
        self.n_epochs = epoch
        if epoch_phase0 is None:
            self.n_epochs_phase0 = int(self.n_epochs * 0.4 + 0.5)
        else:
            self.n_epochs_phase0 = epoch_phase0
        self.every = plot_every
        self.directory = directory
        self.verbose = verbose
        self.mode0_complete = False

        if use_tqdm_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)

        # tensorboard to record and visualize training log (require pytorch 1.1 and up)
        self.writer = SummaryWriter(log_dir=directory)

    def set_input(self, img0, mask, ignore):
        """
        :param img0: input image
        :param mask: CR mask
        :param ignore: loss mask
        :return: None
        """
        self.img0 = Variable(img0.type(self.dtype)).view(-1,1, self.shape, self.shape)
        self.mask = Variable(mask.type(self.dtype)).view(-1,1, self.shape, self.shape)
        self.ignore = 1- (Variable(ignore.type(self.dtype)).view(-1,1, self.shape, self.shape) == 0).type(torch.uint8)

    def validate_mask(self, epoch=None):
        """
        :return: validation loss. print TPR and FPR at threshold = 0.5.
        """
        torch.random.manual_seed(0)
        np.random.seed(0)
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
        if self.verbose:
            print('[TPR=%.3f, FPR=%.3f] @threshold = 0.5' % (TPR, FPR))
        if epoch:
            self.writer.add_scalar('TPR', TPR, epoch)
            self.writer.add_scalar('FPR', FPR, epoch)
            self.writer.add_scalar('validate_loss', lmask, epoch)

        return (lmask)

    def train(self):
        """ call this function to start training network
        :return: None
        """
        if self.verbose:
            print('Begin first {} epochs of training'.format(self.n_epochs_phase0))
            print('Use batch statistics for batch normalization; keep running statistics to be used in phase1')
            print('')
        self.train_phase0(self.n_epochs_phase0)

        filename = self.save()
        self.load(filename)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr/2.5)

        if self.verbose:
            print('Continue onto next {} epochs of training'.format(self.n_epochs - self.n_epochs_phase0))
            print('Batch normalization running statistics frozen and used')
            print('')
        self.train_phase1(self.n_epochs - self.n_epochs_phase0)

    def train_phase0(self, epochs):
        self.network.train()
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every == 0:
                self.plot_example()

            if self.verbose:
                print('----------- epoch = %d -----------' % self.epoch_mask)
            val_loss = self.validate_mask(epoch)
            self.validation_loss.append(val_loss)
            if self.verbose:
                print('loss = %.4f' % (self.validation_loss[-1]))
            if (np.array(self.validation_loss)[-1] == np.array(
                    self.validation_loss).min() and self.epoch_mask > self.save_after):
                filename = self.save()
                if self.verbose:
                    print('Saved to {}.pth'.format(filename))
            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                print('')

    def train_phase1(self, epochs):
        self.set_to_eval()
        self.lr_scheduler._reset()
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every==0:
                self.plot_example()

            if self.verbose:
                print('----------- epoch = %d -----------' % self.epoch_mask)
            valLossMask = self.validate_mask(epoch)
            self.validation_loss.append(valLossMask)
            if self.verbose:
                print('loss = %.4f' % (self.validation_loss[-1]))
            if (np.array(self.validation_loss)[-1] == np.array(
                    self.validation_loss).min() and self.epoch_mask > self.save_after):
                filename = self.save()
                if self.verbose:
                    print('Saved to {}.pth'.format(filename))
            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                print('')

    def plot_example(self):
        plt.figure(figsize=(10, 30))
        plt.subplot(131)
        plt.imshow(np.log(self.img0[0, 0].detach().cpu().numpy()), cmap='gray')
        plt.title('epoch=%d' % self.epoch_mask)
        plt.subplot(132)
        plt.imshow(self.pdt_mask[0, 0].detach().cpu().numpy() > 0.5, cmap='gray')
        plt.title('prediction > 0.5')
        plt.subplot(133)
        plt.imshow(self.mask[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title('ground truth')
        plt.show()

    def set_to_eval(self):
        self.network.eval()

    def optimize_network(self, dat):
        self.set_input(*dat)
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()

    def backward_network(self):
        loss = self.BCELoss(self.pdt_mask * (1 - self.ignore), self.mask * (1 - self.ignore))
        return loss

    def plot_loss(self):
        """ plot validation loss vs. epoch
        :return: None
        """
        plt.figure(figsize=(10,5))
        plt.plot(range(self.epoch_mask), self.validation_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Validation loss')
        plt.show()

    def save(self):
        """ save trained network parameters to date_model_name_epoch*.pth
        :return: None
        """
        time = datetime.datetime.now()
        time = str(time)[:10]
        filename = '%s_%s_epoch%d' % (time, self.name, self.epoch_mask)
        torch.save(self.network.state_dict(), self.directory + filename + '.pth')
        return filename

    def load(self, filename):
        """ Continue training from a previous model state saved to filename
        :param filename: (str) filename (without ".pth") to load model state
        :return: None
        """
        self.network.load_state_dict(torch.load(self.directory + filename + '.pth'))
        loc = filename.find('epoch') + 5
        self.epoch_mask = int(filename[loc:])
