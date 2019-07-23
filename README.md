[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)
[![Documentation Status](https://readthedocs.org/projects/deepcr/badge/?version=latest)](https://deepcr.readthedocs.io/en/latest/?badge=latest)

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Identify and remove cosmic rays from astronomical images using trained convolutional neural networks. Fast on both CPU and GPU.
This is the installable package which implements the methods described in the paper: Zhang & Bloom (2019), submitted.
Code to reproduce benchmarking results in the paper is at: https://github.com/kmzzhang/deepCR-paper
If you use this package, please cite Zhang & Bloom (2019): www.arxiv.org/XXX
This repo is under active development.

<img src="https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg" wdith="90%">

### Installation

```bash
pip install deepCR
```

Or you can install from source:

```bash
git clone https://github.com/profjsb/deepCR.git
cd deepCR/
pip install
```

### Quick Start

With Python >=3.5:

```python
from deepCR import deepCR
from astropy.io import fits
image = fits.getdata("example_flc.fits")
# create an instance of deepCR with specified model configuration
mdl = deepCR(mask="ACS-WFC-F606W-2-32",
	         inpaint="ACS-WFC-F606W-3-32",
             device="CPU")
# apply to input image
# you should examine cleaned_image to verify that proper threshold is used
mask, cleaned_image = mdl.clean(image, threshold = 0.5)

# for quicker mask prediction, skip image inpainting
mask = mdl.clean(image, threshold = 0.5, inpaint=False)

# for probabilistic cosmic ray mask (instead of binary mask)
prob_mask = mdl.clean(image, binary=False)
```

To reduce memory consumption (recommended for image larger than 1k x 1k), deepCR could segment input image to smaller patches of seg*seg, perform CR rejection one by one, and stitch the outputs back to original size. Runtime does not increase by much.
```python
mask, cleaned_image = mdl.clean(image, threshold = 0.5, seg = 256)
mask = mdl.clean(image, threshold = 0.5,  seg = 256)
```
We recommend using patch size (seg) no smaller than 64.

### Currently available models

mask:
    ACS-WFC-F606W-2-4
    ACS-WFC-F606W-2-32(*)

inpaint:
    ACS-WFC-F606W-2-32
    ACS-WFC-F606W-3-32(*)

Recommended models are marked in (*).
Input images should come from _flc.fits files which are in units of electrons.
The two numbers, e.g., 2-32, specifies model hyperparameter. Larger number indicate larger capacity and better performance.

### API Documentation

Documentation is under development at: https://deepcr.readthedocs.io/en/latest/deepCR.html

### Limitations and Caveats

In the current release, the included models have been built and tested only on Hubble Space Telescope (HST) ACS/WFC images in the F606W filter. Applying them to other HST detectors is discouraged. Users should exert caution when applying the models to other filters of ACS/WFC.

### Contributing

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.
