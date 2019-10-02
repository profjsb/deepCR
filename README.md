[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)
[![Documentation Status](https://readthedocs.org/projects/deepcr/badge/?version=latest)](https://deepcr.readthedocs.io/en/latest/?badge=latest) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01651/status.svg)](https://doi.org/10.21105/joss.01651) [![arXiv](https://img.shields.io/badge/astro--ph-1907.09500-blue)](https://arxiv.org/abs/1907.09500) 

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Identify and remove cosmic rays from astronomical images using trained convolutional neural networks.

This package is implements the method described in the paper:
  > [deepCR: Cosmic Ray Rejection with Deep Learning](https://arxiv.org/abs/1907.09500)\
  > Keming Zhang & Joshua Bloom\
  > _arXiv:1907.09500; ApJ in press_
  
If you use this package, please cite the paper above and consider including a
link to this repository.

[Documentation and tutorials](deepcr.readthedocs.io)

[Currently available models](https://deepcr.readthedocs.io/en/latest/model_zoo.html)


<img src="https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg" wdith="90%">

### New for v0.2.0

[DECam](https://deepcr.readthedocs.io/en/latest/model_zoo.html#decam) deepCR model now available!

```python
from deepCR import deepCR
decam_model = deepCR(mask='decam', device='CPU')
```
Note 1: Model is trained on g-band images but is expected to work on 
other filters as well. We are working on benchmarking on different filters 
but before that's done please proceed with caution working with other filters.

Note 1: Inpainting model is TBA for DECam.

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

Quick download of a HST ACS/WFC image

```bash
wget -O jdba2sooq_flc.fits https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/jdba2sooq_flc.fits
```

With Python >=3.5:

For smaller sized images
```python
from deepCR import deepCR
from astropy.io import fits
image = fits.getdata("jdba2sooq_flc.fits")[:512,:512]

# create an instance of deepCR with specified model configuration
mdl = deepCR(mask="ACS-WFC-F606W-2-32",
	     inpaint="ACS-WFC-F606W-2-32",
             device="CPU")

# apply to input image
mask, cleaned_image = mdl.clean(image, threshold = 0.5)
# best threshold is highest value that generate mask covering full extent of CR
# choose threshold by visualizing outputs.
# note that deepCR-inpaint would overestimate if mask does not fully cover CR.

# if you only need CR mask you may skip image inpainting for shorter runtime
mask = mdl.clean(image, threshold = 0.5, inpaint=False)

# if you want probabilistic cosmic ray mask instead of binary mask
prob_mask = mdl.clean(image, binary=False)
```

For WFC full size images (4k * 2k), you should specify **segment = True** to tell deepCR to segment the input image into 256*256 patches, and process one patch at a time.
Otherwise this would take up > 10gb memory. We recommended you use segment = True for images larger than 1k * 1k on CPU. GPU memory limits may be more strict.
```python
image = fits.getdata("jdba2sooq_flc.fits")
mask, cleaned_image = mdl.clean(image, threshold = 0.5, segment = True)
```

(CPU only) In place of **segment = True**, you can also specify **parallel = True** and invoke the multi-threaded version of segment mode. This will speed things up. You don't have to specify segment = True again.
```python
image = fits.getdata("jdba2sooq_flc.fits")
mask, cleaned_image = mdl.clean(image, threshold = 0.5, parallel = True, n_jobs=-1)
```
**n_jobs=-1** makes use of all your CPU cores.

Note that this won't speed things up if you're using GPU!

### Contributing

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.
