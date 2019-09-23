import numpy as np
from tqdm import tqdm as tqdm

from deepCR.util import maskMetric
from deepCR.dataset import dataset, DatasetSim


__all__ = ['roc']


def _roc(model, data, thresholds):
    """ internal function called by roc

    :param model:
    :param data: deepCR.dataset object
    :param thresholds:
    :return: tpr, fpr
    """
    nROC = thresholds.size
    metric = np.zeros((nROC, 4))
    for t in tqdm(range(len(data))):
        dat = data[t]
        pdt_mask = model.clean(dat[0], inpaint=False, binary=False)
        msk = dat[1]
        ignore = dat[2]
        for i in range(nROC):
            binary_mask = np.array(pdt_mask > thresholds[i]) * (1 - ignore)
            metric[i] += maskMetric(binary_mask, msk * (1 - ignore))
    TP, TN, FP, FN = metric[:, 0], metric[:, 1], metric[:, 2], metric[:, 3]
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    return tpr * 100, fpr * 100


def roc(model, image, mask, ignore=None, sky=None, n_mask=1, seed=1, thresholds=np.linspace(0.001, 0.999, 500)):
    """ evaluate model on test set with the ROC curve

    :param model: deepCR object
    :param image: np.ndarray((N, W, H)) image array
    :param mask: np.ndarray((N, W, H)) CR mask array
    :param ignore: np.ndarray((N, W, H)) bad pixel array incl. saturation, etc.
    :param thresholds: np.ndarray(N) FPR grid on which to evaluate ROC curves
    :return: np.ndarray(N), np.ndarray(N): TPR and FPR
    """
    if type(image) == np.ndarray and len(image.shape) == 3:
        data = dataset(image, mask, ignore)
    elif type(image[0]) == str:
        data = DatasetSim(image, mask, sky=sky, n_mask=n_mask, seed=seed)
    else:
        raise TypeError('Input must be numpy data arrays or list of file paths!')

    tpr, fpr = _roc(model, data, thresholds=thresholds)
    return tpr, fpr
