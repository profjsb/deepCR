[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)
[![Documentation Status](https://readthedocs.org/projects/deepcr/badge/?version=latest)](https://deepcr.readthedocs.io/en/latest/?badge=latest)

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Identify and remove cosmic rays from astronomical images using trained convolutional neural networks.

This is the installable package which implements the methods described in the paper: Zhang & Bloom (2019), submitted.

Code to reproduce benchmarking results in the paper is at: https://github.com/kmzzhang/deepCR-paper

If you use this package, please cite Zhang & Bloom (2019): url TBA

<img src="https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg" wdith="90%">

### Installation

```bash
pip install deepCR
```

Or you can install from source:

```bash
git clone https://github.com/profjsb/deepCR.git
cd deepCR/
python setup.py install
```

### Quick Start

With Python >=3.5:

For smaller sized images
```python
from deepCR import deepCR
from astropy.io import fits
image = fits.getdata("example_flc.fits")[:512,:512]

# create an instance of deepCR with specified model configuration
mdl = deepCR(mask="ACS-WFC-F606W-2-32",
	     inpaint="ACS-WFC-F606W-3-32",
             device="CPU")

# apply to input image
mask, cleaned_image = mdl.clean(image, threshold = 0.5)
# visualize those outputs to choose an adequate threshold
# note that deepCR-inpaint would be inaccurate if mask does not fully cover CR

# if you only need CR mask you may skip image inpainting for shorter runtime
mask = mdl.clean(image, threshold = 0.5, inpaint=False)

# if you want probabilistic cosmic ray mask instead of binary mask
prob_mask = mdl.clean(image, binary=False)
```

For WFC full size images (4k * 2k), you should specify *segment = True* to tell deepCR to segment the input image into 256*256 patches, and process one patch at a time.
Otherwise this would take up > 10gb memory. We recommended you use segment = True for images larger than 1k * 1k on CPU. GPU memory limits may be more strict.
```python
mask, cleaned_image = mdl.clean(image, threshold = 0.5, segment = True)
mask = mdl.clean(image, threshold = 0.5, segment = True)
```

(CPU only) In place of segment = True, you can also specify *parallel = True* and invoke the multi-threaded version of *segment = True*. This will speed things up. You don't have to specify segment = True again.
```python
mask, cleaned_image = mdl.clean(image, threshold = 0.5, parallel = True, n_jobs=-1)
mask = mdl.clean(image, threshold = 0.5, parallel = True, n_jobs=-1)
```
n_jobs=-1 makes use of all your CPU cores.

Note that this won't speed things up if you're using GPU!

### Currently available models

mask:

    ACS-WFC-F606W-2-4

    ACS-WFC-F606W-2-32(*)

inpaint:

    ACS-WFC-F606W-2-32

    ACS-WFC-F606W-3-32(*)

Recommended models are marked in (*). Larger number indicate larger capacity and better performance.

Input images should come from *_flc.fits* files which are in units of electrons.


### API Documentation

Full documentation is under development at: https://deepcr.readthedocs.io/en/latest/deepCR.html

### Limitations and Caveats

In the current release, the included models have been trained and tested only on Hubble Space Telescope (HST) ACS/WFC images in the F606W filter. They may work well on nearby ACS/WFC filters, though users should exert caution.

The ACS/WFC models are not expected to work optimally on other HST detectors, though we'd be interested to know if you find additional use cases for them.

### Contributing

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.
