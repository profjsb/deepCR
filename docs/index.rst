.. deepCR documentation master file, created by
   sphinx-quickstart on Mon Jul 22 20:23:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images
======================================================================

.. image:: https://travis-ci.com/profjsb/deepCR.svg?token=baKtC9yCzzwzzqM9ihAX&branch=master
   :target: https://travis-ci.com/profjsb/deepCR
   :alt: Build Status

.. image:: https://codecov.io/gh/profjsb/deepCR/branch/master/graph/badge.svg?token=SIwJFmKJqr
   :target: https://codecov.io/gh/profjsb/deepCR
   :alt: codecov

.. image:: https://readthedocs.org/projects/deepcr/badge/?version=latest
:target: https://deepcr.readthedocs.io/en/latest/?badge=latest
:alt: Documentation Status

Welcome to the documentation for `deepCR`. You will use `deepCR` to apply a learned convolutional neural net (CNN) model to a 2d ``numpy`` array to identify and remove cosmic rays, on multi-core CPUs or GPUs.

.. image:: https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg


Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install deepCR

Or you can install from source:

.. code-block:: bash

   git clone https://github.com/profjsb/deepCR.git
   cd deepCR/
   pip install

Currently available models
^^^^^^^^^^^

mask:

    ACS-WFC-F606W-2-4

    ACS-WFC-F606W-2-32(*)

inpaint:

    ACS-WFC-F606W-2-32(*)

    ACS-WFC-F606W-3-32

Recommended models are marked in (*). Larger number indicate larger capacity.

Note that trained models may have input unit or preprocessing requirements. For the ACS-WFC-F606W models, input images must come from *_flc.fits* files which are in units of electrons.


Limitations and Caveats
^^^^^^^^^^^^^^^^^^^^^^^

In the current release, the included models have been built and tested only on Hubble Space Telescope (HST) ACS/WFC images in the F606W filter. Application to native-spatial resolution (ie. not drizzled), calibrated images from ACS/F606W (\ ``*_flc.fits``\ ) is expected to work well. Use of these prepackaged models in other observing modes with HST or spectroscopy is not encouraged. We are planning hosting a "model zoo" that would allow deepCR to be adapted to a wide range of instrument configurations.

Contributing
^^^^^^^^^^^^

We are very interested in getting bug fixes, new functionality, and new trained models from the community (especially for ground-based imaging and spectroscopy). Please fork this repo and issue a PR with your changes. It will be especially helpful if you add some tests for your changes.

If you run into any issues, please don't hesitate to `open an issue on GitHub
<https://github.com/profjsb/deepCR/issues>`_.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   tutorial_use
   tutorial_train
   model_zoo
   deepCR
