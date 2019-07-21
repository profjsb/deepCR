[![Build Status](https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master)](https://travis-ci.com/profjsb/deepCR) [![codecov](https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr)](https://codecov.io/gh/profjsb/deepCR)

## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Apply a learned convolutional neural net (CNN) model to a 2d `numpy` array to identify and remove cosmic rays, on multi-core CPUs or GPUs.

<img src="imgs/postage-sm.jpg" wdith="90%">


This is the installable package which implements the methods described in the paper: Zhang & Bloom (2019), submitted. All figures and speed/scoring benchmarks relative to existing solutions can be found in the paper Github repo: https://github.com/kmzzhang/deepCR-paper

### Quick Start

With Python >=3.5:

```python
from deepCR import model
mdl = model.deepCR(mask="ACS-WFC-F606W-2-4",
	               inpaint="ACS-WFC-F606W-2-32",
                   device="GPU")
mask, cleaned_image = mdl.clean(image)
```

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

### Limitations and Caveats

In the current release, the included models have been built and tested only on ACS Hubble Space Telescope (HST) images in the F606W filter. Application to native-resolution flattened images from ACS/F606W (`*_flt.fits`) should work well. Use of these prepackaged models in other observing modes with HST, ground-based images, or spectroscopy is not encouraged. However, we expect the method implemented herein to work on such data with sufficient training data.

You will need to run your model on arrays which have the same bite ordering as the native machine. Some FITS readers do not switch bit order automatically. In this case, you can ask `numpy` to switch. For instance,

```python
out = mdl.clean(my_hst_image.astype("<f4"))
```

### Contributing

We are very interested in getting bug fixes, new functionality, and new models from the community (built especially on ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes. 