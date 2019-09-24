Model zoo: a collection of models with benchmarks
======================================================================

Hubble ACS/WFC
^^^^^^^^^^^^^^
.. code-block:: python

    from deepCR import deepCR
    model_cpu = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    model_gpu = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='GPU')


Input images for the ACS/WFC model should come from *_flc.fits* files which are in units of electrons.
The current ACS/WFC model is trained and benchmarked on HST ACS/WFC images in the F606W filter.
Visual inspection shows that these models also work well on filters from F435W to F814W. However, users should use a
higher threshold for images of denser fields in short wavelength filters to minimize false detections, if any.

All filter WFC model on the way.

.. image:: https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/acs_wfc_f606w_roc.png


DECam
^^^^^
DECam is a high-performance, wide-field CCD imager mounted at the prime focus of the Blanco 4-m telescope at CTIO.

.. code-block:: python

    from deepCR import deepCR
    model_cpu = deepCR(mask='decam', device='CPU')
    model_gpu = deepCR(mask='decam', device='GPU')

.. image:: https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/decam.png

Input images should come from calibrated images in the original unit (adu).
The ROC curves above are produced from a test set that contains noise in cosmic ray labels.
This causes TPR to be lower than actual because the deepCR predicted CR labels is essentially noise-free.

Generic Ground Imaging
^^^^^
Coming soon...
