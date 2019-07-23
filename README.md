[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)
[![Documentation Status](https://readthedocs.org/projects/deepcr/badge/?version=latest)](https://deepcr.readthedocs.io/en/latest/?badge=latest)

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Apply a learned convolutional neural net (CNN) model to a 2d `numpy` array to identify and remove cosmic rays, on multi-core CPUs or GPUs.

<img src="https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg" wdith="90%">


This is the installable package which implements the methods described in the paper: Zhang & Bloom (2019), submitted. Code to benchmark the model and to generate figures and tables in the paper can be found in the deepCR-paper Github repo: https://github.com/kmzzhang/deepCR-paper

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

image = fits.getdata("*********_flc.fits")
mdl = deepCR(mask="ACS-WFC-F606W-2-32",
	     inpaint="ACS-WFC-F606W-2-32",
             device="GPU")
mask, cleaned_image = mdl.clean(image, threshold = 0.5)
```
Note:
Input image must be in units of electrons

To reduce memory consumption (recommended for image larger than 1k x 1k):
```python
mask, cleaned_image = mdl.clean(image, threshold = 0.5, seg = 256)
```
which segments the input image into patches of 256*256, seperately perform CR rejection on the patches, before stitching back to original image size.

### Currently available models

mask:
    ACS-WFC-F606W-2-4
    ACS-WFC-F606W-2-32(*)

inpaint:
    ACS-WFC-F606W-2-32
    ACS-WFC-F606W-3-32(*)

The two numbers following instrument configuration specifies model size, with larger number indicating better performing model at the expense of runtime. Recommanded models are marked in (*). For benchmarking of these models, please refer to the original paper.

### API Documentation

Documentation is under development at: https://deepcr.readthedocs.io/en/latest/deepCR.html

### Limitations and Caveats

In the current release, the included models have been built and tested only on Hubble Space Telescope (HST) ACS/WFC images in the F606W filter. Application to native-spatial resolution (ie. not drizzled), calibrated images from ACS/F606W (`*_flc.fits`) is expected to work well. Use of these prepackaged models in other observing modes with HST or spectroscopy is not encouraged. We are planning hosting a "model zoo" that would allow deepCR to be adapted to a wide range of instrument configurations.

### Contributing

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.

