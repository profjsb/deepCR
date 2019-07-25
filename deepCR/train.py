"""main module to instantiate deepCR models and use them
"""

import torch
from tqdm import tqdm_notebook as tqdm


from deepCR.unet import WrappedModel, UNet2Sigmoid
from learned_models import mask_dict, inpaint_dict, default_model_path

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.tensor as tensor
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

    def __init__(self, image, mask, use=None, name='model', hidden=32, gpu=False, epoch=50, batch_size=16, lr=0.005, verbose=True, save_after=0, every=10):

        if use is None:
            use = np.zeros_like(image)
        data_train = dataset(image, mask, use, part='train')
        data_val = dataset(image, mask, use, part='val')
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=False, num_workers=1)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=1)
        self.name = name
        self.verbose = verbose

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
        self.lossMask_train = []
        self.lossMask_val = []
        self.itr = 0
        self.epoch_mask = 0
        self.save_after = save_after
        self.n_epochs = epoch
        self.every = every

    def set_input(self, img0, mask, use):
        self.img0 = Variable(img0.type(self.dtype)).view(-1, 1, 256, 256)
        self.mask = Variable(mask.type(self.dtype)).view(-1, 1, 256, 256)
        self.use = Variable(use.type(self.dtype)).view(-1, 1, 256, 256)

    def validate_mask(self):
        lmask = 0; count = 0
        metric = np.zeros(4)
        for i, dat in enumerate(self.TrainLoader):
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            loss = self.backward_network()
            lmask += float(loss.detach()) * n
            metric += maskMetric(self.pdt_mask.reshape(-1, 256, 256).detach().cpu().numpy() > 0.1, dat[1].numpy())
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
        print(TP, TN, FP, FN)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        print('TPR=%.3f   FPR=%.3f' % (TPR * 100, FPR * 100))
        return (lmask)
    
    def train(self):
        self.network.train()
        for epoch in tqdm(range(self.n_epochs)):
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
        self.itr += self.img0.shape[0]
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()
        self.lossMask_train.append(float(loss.detach()))

    def backward_network(self):
        loss = self.BCELoss(self.pdt_mask * (1 - self.use), self.mask * (1 - self.use))
        return (loss)

    def save_mask(self):
        time = datetime.datetime.now()
        time = str(time)[:10]
        filename = '%s_%s_epoch%d' % (time, self.name, self.epoch_mask)
        print(filename)
        torch.save(self.network.state_dict(), filename + '.pth')
        lossTrain = np.array(self.lossMask_train)
        lossVal = np.array(self.lossMask_val)
        np.save(filename + '_mask_train.npy', lossTrain)
        np.save(filename + '_mask_val.npy', lossVal)

    def load_mask(self, filename):
        # self.load_model_mask(filename)
        self.network.load_state_dict(torch.load(filename + '.pth'))
        # self.load_loss_mask(filename)
        self.lossMask_train = list(np.load(filename + '_mask_train.npy'))
        self.lossMask_val = list(np.load(filename + '_mask_val.npy'))
        loc = filename.find('epoch') + 5
        self.epoch_mask = int(filename[loc:])