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

Identify and remove cosmic rays from astronomical images using trained convolutional neural networks.

.. image:: https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg


Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install deepCR

Or you can install from source:

.. code-block:: bash

   git clone https://github.com/profjsb/deepCR.git
   cd deepCR/
   python setup.py install

Currently available models
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from deepCR import deepCR
   decam_model = deepCR(mask='decam', device='CPU')
   acswfc_model = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='GPU')
```

For detailed descriptions and requirements of currently available models, please visit the model_zoo page below.

If you run into any issues, please don't hesitate to `open an issue on GitHub
<https://github.com/profjsb/deepCR/issues>`_.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   tutorial_use
   tutorial_train
   model_zoo
   deepCR
