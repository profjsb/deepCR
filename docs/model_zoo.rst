Model zoo: a collection of models with benchmarks
======================================================================

Hubble ACS/WFC
^^^^^^^^^^^^^^
.. code-block:: python

    from deepCR import deepCR
    model_cpu = deepCR(mask='ACS-WFC-2-32', inpaint='ACS-WFC-F606W-2-32', device='CPU')
    model_gpu = deepCR(mask='ACS-WFC-2-32', inpaint='ACS-WFC-F606W-2-32', device='GPU')

``ACS-WFC-2-32`` model is trained with image sets from ACS/WFC F435W, F606W, F814W. Individual models for each filter were also trained to demonstrate that the global model can be used for all three filter images. Refer to the ROC curves below. The Y-axis denotes the true positive rate (TPR), and X-axis denotes the false positive rate (FPR). 

Since the training and test datasets comprise images from three different fields (extragalactic field, globular cluster, resolved galaxy), the models were tested for each field. As shown, there is no significant discrepancy between individual models' performance and the global model. 

.. image:: https://raw.githubusercontent.com/kgb0255/deepCR_james/master/imgs/acs_wfc.f435w_test.png

.. image:: https://raw.githubusercontent.com/kgb0255/deepCR_james/master/imgs/acs_wfc.f606w_test.png

.. image:: https://raw.githubusercontent.com/kgb0255/deepCR_james/master/imgs/acs_wfc.f814w_test.png


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

**Note 0**: The current DECam model as preliminary as it is trained on median-coadd science images which
may cause false positives of stars in single frame images in some cases.

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
