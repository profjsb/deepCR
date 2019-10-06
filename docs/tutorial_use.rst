Quickstart: Using deepCR
==============

Quick download of a HST ACS/WFC image

.. code-block:: bash
    wget -O jdba2sooq_flc.fits https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/jdba2sooq_flc.fits

With Python >=3.5:

For smaller sized images (smaller than ~1Mpix)
.. code-block:: python

    from deepCR import deepCR
    from astropy.io import fits
    image = fits.getdata("jdba2sooq_flc.fits")[:512,:512]

# create an instance of deepCR with specified model configuration
.. code-block:: python

    mdl = deepCR(mask="ACS-WFC-F606W-2-32",
             inpaint="ACS-WFC-F606W-2-32",
                 device="CPU")

# apply to input image
.. code-block:: python

    mask, cleaned_image = mdl.clean(image, threshold = 0.5)

# best threshold is highest value that generate mask covering full extent of CR
# choose threshold by visualizing outputs.

# if you only need CR mask you may skip image inpainting and save time
mask = mdl.clean(image, threshold = 0.5, inpaint=False)

# if you want probabilistic cosmic ray mask instead of binary mask
.. code-block:: python

    prob_mask = mdl.clean(image, binary=False)

There's also the option to segment your input image into smaller pieces (default: 256-by-256)
and process the individual piece seperately before stitching them back together. This enables
multi-process parallelism and saves memory.

Segment-and-stitching is enabled by **n_jobs>1**, which specified the number of processes to utilize.
**n_jobs=-1** is the number of available virtual cores on your machine and is optimized for time
when your torch is not intel MKL optimized (see below for more details).
.. code-block:: python

    image = fits.getdata("jdba2sooq_flc.fits")
    mask, cleaned_image = mdl.clean(image, threshold = 0.5, n_jobs=-1)


If your torch is intel MKL optimized, it's not necessary to open up many processes and one process
should utilize half of the CPUs available. Monitor CPU usage -- if CPU usage for single process
is > 100% it means intel MKL is in place. In this case, ** n_jobs<=4** is advised.

For single process segment-and-stitching, you need to manually enable **segment = True** because
the default **n_jobs=1** assumes **segment = False**.
.. code-block:: python

    image = fits.getdata("jdba2sooq_flc.fits")
    mask, cleaned_image = mdl.clean(image, threshold = 0.5, segment = True)

Note that this won't speed things up if you're using GPU!
