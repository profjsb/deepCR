"""main module to instantiate deepCR models and use them
"""

import numpy as np
import torch
import torch.nn as nn
from torch import from_numpy
from tqdm import tqdm


from deepCR.unet import WrappedModel, UNet2Sigmoid
from learned_models import mask_dict, inpaint_dict, default_model_path

import torch.nn.functional as F
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

    def __init__(self, image, mask, use=None, name='model', hidden=32, gpu=False, epoch=50, batch_size=16, lr=0.01, verbose=True, save_after=0):

        if use is None:
            use = np.zeros_like(image)
        data_train = dataset(image, mask, use, part='train')
        data_val = dataset(image, mask, use, part='val')
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=1)
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

    def set_input(self, img0, mask, use):
        self.img0 = Variable(img0.type(self.dtype)).view(-1, 1, 256, 256)
        self.mask = Variable(mask.type(self.dtype)).view(-1, 1, 256, 256)
        self.use = Variable(use.type(self.dtype)).view(-1, 1, 256, 256)

    def validate_mask(self):
        lmask = 0; count = 0
        metric = np.zeros(4)
        for i, dat in enumerate(self.ValLoader):
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            loss = self.backward_network()
            lmask += float(loss.detach()) * n
            metric += maskMetric(self.pdt_mask.reshape(-1, 256, 256).detach().cpu().numpy() > 0.5, dat[2].numpy())
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
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
        np.save('model/loss/' + filename + '_mask_train.npy', lossTrain)
        np.save('model/loss/' + filename + '_mask_val.npy', lossVal)

    def load_mask(self, filename):
        # self.load_model_mask(filename)
        self.network.load_state_dict(torch.load('model/' + filename + '_network.pth'))
        # self.load_loss_mask(filename)
        self.lossMask_train = list(np.load('model/loss/' + filename + '_mask_train.npy'))
        self.lossMask_val = list(np.load('model/loss/' + filename + '_mask_val.npy'))
        loc = filename.find('epoch') + 5
        self.epoch_mask = int(filename[loc:])