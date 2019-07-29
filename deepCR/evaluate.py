import numpy as np
from deepCR.util import maskMetric
from deepCR.dataset import dataset
from tqdm import tqdm_notebook as tqdm

__all__ = 'roc'


def _roc(model, data, thresholds):
    """ internal function called by roc

    :param model:
    :param data: deepCR.dataset object
    :param thresholds:
    :return: tpr, fpr
    """
    nROC = thresholds.size
    metric = np.zeros((nROC, 4))
    for t in range(len(data)):
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


def roc(model, image, mask, ignore=None, thresholds=np.linspace(0.001, 0.999, 500)):
    """ evaluate model on test set with the ROC curve

    :param model: deepCR object
    :param image: np.ndarray((N, W, H)) image array
    :param mask: np.ndarray((N, W, H)) CR mask array
    :param ignore: np.ndarray((N, W, H)) bad pixel array incl. saturation, etc.
    :param thresholds: np.ndarray(N) FPR grid on which to evaluate ROC curves
    :return: np.ndarray(N), np.ndarray(N): TPR and FPR
    """
    data = dataset(image=image, mask=mask, ignore=ignore)
    tpr, fpr = _roc(model, data, thresholds=thresholds)
    return tpr, fpr
