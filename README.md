[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)
[![Documentation Status](https://readthedocs.org/projects/deepcr/badge/?version=latest)](https://deepcr.readthedocs.io/en/latest/?badge=latest) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01651/status.svg)](https://doi.org/10.21105/joss.01651) [![arXiv](https://img.shields.io/badge/astro--ph-1907.09500-blue)](https://arxiv.org/abs/1907.09500) 

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Identify and remove cosmic rays from astronomical images using trained convolutional neural networks.

This package is implements the method described in the paper:
  > **deepCR: Cosmic Ray Rejection with Deep Learning**\
  > Keming Zhang & Joshua Bloom 2020\
  > _[Published in the Astrophysical Journal](https://iopscience.iop.org/article/10.3847/1538-4357/ab3fa6)\
  [arXiv:1907.09500](https://arxiv.org/abs/1907.09500)_
  
If you use this package, please cite the paper above and consider including a
link to this repository.

[Documentation and tutorials](https://deepcr.readthedocs.io)

[Currently available models](https://deepcr.readthedocs.io/en/latest/model_zoo.html)


<img src="https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg" wdith="90%">


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

Quick download of a HST ACS/WFC image

```bash
wget -O jdba2sooq_flc.fits https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/jdba2sooq_flc.fits
```

With Python >=3.5:

For smaller sized images (smaller than ~1Mpix)
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

# if you only need CR mask you may skip image inpainting and save time
mask = mdl.clean(image, threshold = 0.5, inpaint=False)

# if you want probabilistic cosmic ray mask instead of binary mask
prob_mask = mdl.clean(image, binary=False)
```

There's also the option to segment your input image into smaller pieces (default: 256-by-256)
and process the individual piece seperately before stitching them back together. This enables
multi-process parallelism and saves memory.

Segment-and-stitching is enabled by **n_jobs>1**, which specified the number of processes to utilize.
**n_jobs=-1** is the number of available virtual cores on your machine and is optimized for time
when your torch is not intel MKL optimized (see below for more details). 
```python
image = fits.getdata("jdba2sooq_flc.fits")
mask, cleaned_image = mdl.clean(image, threshold = 0.5, n_jobs=-1)

```
If your torch is intel MKL optimized, it's not necessary to open up many processes and one process
should utilize half of the CPUs available. Monitor CPU usage -- if CPU usage for single process 
is > 100% it means intel MKL is in place. In this case, ** n_jobs<=4** is advised. 

For single process segment-and-stitching, you need to manually enable **segment = True** because 
the default **n_jobs=1** assumes **segment = False**.
```python
image = fits.getdata("jdba2sooq_flc.fits")
mask, cleaned_image = mdl.clean(image, threshold = 0.5, segment = True)
```

Note that this won't speed things up if you're using GPU!

### Contributing

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.
