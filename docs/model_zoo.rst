Model zoo: a collection of models with benchmarks
======================================================================

Hubble ACS/WFC
^^^^^^^^^^^^^^
.. code-block:: python

    from deepCR import deepCR
    model_cpu = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    model_gpu = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-2-32', device='GPU')

This model is trained on the f606w filter. Visual inspection shows that it also work well on sparser images taken in
filters from F435W to F814W. However, there may be increased false detections in dense stellar fields for
filters < f606w.

Input images for the ACS/WFC model should come from *_flc.fits* files which are in units of electrons.

We are currently working on a generic HST model that works with most imagers and filters.

.. image:: https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/acs_wfc_f606w_roc.png


DECam
^^^^^
DECam is a high-performance, wide-field CCD imager mounted at the prime focus of the Blanco 4-m telescope at CTIO.

.. code-block:: python

    from deepCR import deepCR
    model_cpu = deepCR(mask='decam', device='CPU')
    model_gpu = deepCR(mask='decam', device='GPU')

.. image:: https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/decam_v1.png

The ROC curves above are produced from a test set that contains noise in cosmic ray labels.
This causes TPR to be lower than actual because the deepCR predicted CR labels is essentially noise-free.

Note 1: Output will include some bad pixels and columns, and sometimes blooming patterns.
In particular, the blooming artifacts in the CR mask might be 1-2 pixels larger than the
actual blooming size and cannot be excluded by subtracting by a saturation mask.

Note 2:Input images should come from calibrated images in the original unit (adu).

Note 3: Model is trained on g-band images but is expected to work on
other filters as well. We have benchmarked on g-band and r-band and are working i-band and z-band
but only expect minor differences from the ROC curves above.

Note 4: For extremely short or long exposures (t_exp < 10s or t_exp > 1800s), please visually verify mask output.

Generic Ground Imaging
^^^^^^^^^^^^^^^^^^^^^^
Coming soon...
