import numpy as np
from tqdm import tqdm as tqdm

from deepCR.util import maskMetric
from deepCR.dataset import dataset, DatasetSim
from skimage.morphology import disk, dilation
import astroscrappy.astroscrappy as lac


__all__ = ['roc', 'roc_lacosmic']


def _roc_lacosmic(data, sigclip, objlim=2, dilate=None, gain=1):
    nROC = sigclip.size
    metric = [np.zeros((nROC, 4)), np.zeros((nROC, 4))]
    for t in tqdm(range(len(data))):
        dat = data[t]
        image = dat[0]
        msk = dat[1]
        ignore = dat[2]
        for i in range(nROC):
            pdt_mask, cleanArr = lac.detect_cosmics(image, sigclip=sigclip[i], sigfrac=0.3, objlim=objlim,
                                                    gain=gain, readnoise=5, satlevel=np.inf, sepmed=False,
                                                    cleantype='medmask', niter=4)
            pdt_mask *= (1 - image[3]).astype(bool)
            metric[0][i] += maskMetric(pdt_mask, msk * (1 - ignore))
            if dilate is not None:
                pdt_mask = dilation(pdt_mask, dilate)
                metric[1][i] += maskMetric(pdt_mask, msk * (1 - ignore))
    TP, TN, FP, FN = metric[0][:, 0], metric[0][:, 1], metric[0][:, 2], metric[0][:, 3]
    TP1, TN1, FP1, FN1 = metric[1][:, 0], metric[1][:, 1], metric[1][:, 2], metric[1][:, 3]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TPR1 = TP1 / (TP1 + FN1)
    FPR1 = FP1 / (FP1 + TN1)

    return (TPR * 100, FPR * 100), (TPR1 * 100, FPR1 * 100)


def _roc(model, data, thresholds, dilate=None):
    """ internal function called by roc

    :param model:
    :param data: deepCR.dataset object
    :param thresholds:
    :return: tpr, fpr
    """
    nROC = thresholds.size
    metric = [np.zeros((nROC, 4)), np.zeros((nROC, 4))]
    for t in tqdm(range(len(data))):
        dat = data[t]
        pdt_mask = model.clean(dat[0], inpaint=False, binary=False)
        msk = dat[1]
        ignore = dat[2]
        for i in range(nROC):
            binary_mask = np.array(pdt_mask > thresholds[i]) * (1 - ignore)
            metric[0][i] += maskMetric(binary_mask, msk * (1 - ignore))
            if dilate is not None:
                binary_mask = dilation(binary_mask, dilate)
                metric[1][i] += maskMetric(binary_mask, msk * (1 - ignore))

    TP, TN, FP, FN = metric[0][:, 0], metric[0][:, 1], metric[0][:, 2], metric[0][:, 3]
    TP1, TN1, FP1, FN1 = metric[1][:, 0], metric[1][:, 1], metric[1][:, 2], metric[1][:, 3]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TPR1 = TP1 / (TP1 + FN1)
    FPR1 = FP1 / (FP1 + TN1)

    return (TPR * 100, FPR * 100), (TPR1 * 100, FPR1 * 100)


def roc(model, image, mask, ignore=None, sky=None, n_mask=1, seed=1, thresholds=np.linspace(0.001, 0.999, 500),
        dilate=False, rad=1):
    """ evaluate model on test set with the ROC curve

    :param model: deepCR object
    :param image: np.ndarray((N, W, H)) image array
    :param mask: np.ndarray((N, W, H)) CR mask array
    :param ignore: np.ndarray((N, W, H)) bad pixel array incl. saturation, etc.
    :param thresholds: np.ndarray(N) FPR grid on which to evaluate ROC curves
    :return: np.ndarray(N), np.ndarray(N): TPR and FPR
    """
    kernel = None
    if dilate:
        kernel = disk(rad)
    if type(image) == np.ndarray and len(image.shape) == 3:
        data = dataset(image, mask, ignore)
    elif type(image[0]) == str:
        data = DatasetSim(image, mask, sky=sky, n_mask=n_mask, seed=seed)
    else:
        raise TypeError('Input must be numpy data arrays or list of file paths!')
    (tpr, fpr), (tpr_dilate, fpr_dilate) = _roc(model, data, thresholds=thresholds, dilate=kernel)
    if dilate:
        return (tpr, fpr), (tpr_dilate, fpr_dilate)
    else:
        return tpr, fpr


def roc_lacosmic(image, mask, sigclip, ignore=None, sky=None, n_mask=1, seed=1, objlim=2, gain=1,
        dilate=False, rad=1):
    """ evaluate model on test set with the ROC curve

    :param model: deepCR object
    :param image: np.ndarray((N, W, H)) image array
    :param mask: np.ndarray((N, W, H)) CR mask array
    :param ignore: np.ndarray((N, W, H)) bad pixel array incl. saturation, etc.
    :param thresholds: np.ndarray(N) FPR grid on which to evaluate ROC curves
    :return: np.ndarray(N), np.ndarray(N): TPR and FPR
    """
    kernel = None
    if dilate:
        kernel = disk(rad)
    if type(image) == np.ndarray and len(image.shape) == 3:
        data = dataset(image, mask, ignore)
    elif type(image[0]) == str:
        data = DatasetSim(image, mask, sky=sky, n_mask=n_mask, seed=seed)
    else:
        raise TypeError('Input must be numpy data arrays or list of file paths!')
    (tpr, fpr), (tpr_dilate, fpr_dilate) = _roc_lacosmic(data, sigclip=sigclip, objlim=objlim, dilate=kernel, gain=gain)
    if dilate:
        return (tpr, fpr), (tpr_dilate, fpr_dilate)
    else:
        return tpr, fpr
