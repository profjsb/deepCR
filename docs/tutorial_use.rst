Quickstart: Using deepCR
==============

Quick download of a HST ACS/WFC image

.. code-block:: bash

    wget -O jdba2sooq_flc.fits https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/jdba2sooq_flc.fits

For smaller sized images

.. code-block:: python

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

For WFC full size images (4k * 2k), you should specify **segment = True** to tell deepCR to segment the input image into 256*256 patches, and process one patch at a time.
Otherwise this would take up > 10gb memory. We recommended you use segment = True for images larger than 1k * 1k on CPU. GPU memory limits may be more strict.

.. code-block:: python

    image = fits.getdata("jdba2sooq_flc.fits")
    mask, cleaned_image = mdl.clean(image, threshold = 0.5, segment = True)

(CPU only) In place of **segment = True**, you can also specify **parallel = True** and invoke the multi-threaded version of segment mode (**segment = True**). This will speed things up a lot. You don't need to specify **segment = True** again.

.. code-block:: python

    image = fits.getdata("jdba2sooq_flc.fits")
    mask, cleaned_image = mdl.clean(image, threshold = 0.5, parallel = True, n_jobs=-1)

**n_jobs=-1** makes use of all your CPU cores.

Note that this won't speed things up if you're using GPU!
