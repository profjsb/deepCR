[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)
[![Documentation Status](https://readthedocs.org/projects/deepcr/badge/?version=latest)](https://deepcr.readthedocs.io/en/latest/?badge=latest) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01651/status.svg)](https://doi.org/10.21105/joss.01651) [![arXiv](https://img.shields.io/badge/astro--ph-1907.09500-blue)](https://arxiv.org/abs/1907.09500) 

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Identify and remove cosmic rays from astronomical images using trained convolutional neural networks.
Currently supports Hubble Space Telescope ACS-WFC and WFC3-UVIS cameras. 

This package is implements the method described in the paper:
  > **deepCR: Cosmic Ray Rejection with Deep Learning**\
  > Keming Zhang & Joshua Bloom 2020\
  > _[Published in the Astrophysical Journal](https://iopscience.iop.org/article/10.3847/1538-4357/ab3fa6)\
  [arXiv:1907.09500](https://arxiv.org/abs/1907.09500)_

<img src="https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg" wdith="90%">

[Documentation and tutorials](https://deepcr.readthedocs.io)

### Currently Available Models
```
ACS-WFC: Supports all filters on HST ACS-WFC. Reference: [Kwon, Zhang & Bloom 2021](https://iopscience.iop.org/article/10.3847/2515-5172/abf6c8/meta).
WFC3-UVIS: Supports all filters on WFC3-UVIS. Reference: [Chen et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad1602/meta) 
```

### Installation

```bash
pip install deepCR
```

Or you can install from source:

```bash
git clone https://github.com/profjsb/deepCR.git
cd deepCR/
pip install .
```

### Quick Start

Quick download of a HST ACS/WFC and a WFC3/UVIS image

```bash
wget -O jdba2sooq_flc.fits https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/jdba2sooq_flc.fits
wget -O ietx1ab1q_flc.fits https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/ietx1ab1q_flc.fits
```

```python
from deepCR import deepCR
from astropy.io import fits
image = fits.getdata("jdba2sooq_flc.fits")[:512,:512]

# Create an instance of deepCR for ACS-WFC
mdl = deepCR(mask="ACS-WFC")
# mdl = deepCR(mask="WFC3-UVIS") for WFC3-UVIS

# Apply the model
mask = mdl.clean(image, threshold = 0.5)
# 0.5 threshold usually works for ACS/WFC
# 0.1-0.2 for WFC3/UVIS (see Chen et al. 2024)

# Probabilistic mask could be helpful in determining threshold
prob_mask = mdl.clean(image, binary=False)

# Optional inpainting with median filtering
mask, cleaned_image = mdl.clean(image, threshold = 0.5, inpaint=True)
```

For larger images you may want to enable ``mdl.clean(..., segment=True, patch=512)`` option to prevent memory
overflow. This option segment your input image into small squares of 512 by 512 for input into the model,
where the CR masks are stitched back together. In this case,
you may also enable multiprocessing by specifying ``n_jobs>1``. Note that this won't speed things up
if you're using GPU!

### Contributing

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.
