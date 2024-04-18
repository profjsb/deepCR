Quickstart: Training new deepCR models
==============

Dataset construction
^^^^^^^^^^^^^^^^^^^^


Training new models
^^^^^^^^^^^^^^^^^^^

deepCR supports two type of training data:

**Type 1: Paired data of image -- mask. Stored in one huge npy file (this will be deprecated)**

image: np.ndarray (N,W,W). Array containing N input images chucks of W*W

mask: np.ndarray (N,W,W). Array containing N ground truth CR mask chucks of W*W.

ignore: (optional) np.ndarray (N,W,W). Array containing flags where we do not want to train or evaluate the model on. This
typically includes bad pixels and saturations, or any other artifact falsely included in ``mask``

sky: (optional) np.ndarray (N,) Array containing sky background level for each image chunks.

.. code-block:: python

    from deepCR import train
    trainer = train(image, mask, ignore=ignore, sky=sky, aug_sky=[-0.9, 3], name='mymodel', gpu=True, epoch=50,
    save_after=20, plot_every=10, use_tqdm=False)
    trainer.train()
    filename = trainer.save() # not necessary if save_after is specified


**Type 2: Paired data of image -- mask. Data passed along as list of paths of single images**

image: List of paths to image segments of shape (2or3,W,W). The three dimensions are image, CR mask, (optional) ignore mask.

To provide the sky background level, save image chunks from the same exposure in the same sub-directory, and include the
sky background level in sky.npy under that directory, e.g., /data/image0/sky.npy where the image segments are to be saved
in the same directory.

Remember not to include sky.npy in the list of training images especially when traversing your directory.

.. code-block:: python

    from deepCR import train
    trainer = train(image, mode='pair', aug_sky=[-0.9, 3], name='mymodel', gpu=True, epoch=50,
    save_after=20, plot_every=10, use_tqdm=False)
    trainer.train()
    filename = trainer.save() # not necessary if save_after is specified


**Type 2: Simulate CR affected images from clean images and CR in dark frames**

image: list. List containing complete paths to input images stored in *.npy of shape (W, W)

mask: list. List containing complete paths to cosmic rays stored in *.npy of shape (2, W, W). mask[0] is cr image
and mask[1] is cr mask.

.. code-block:: python

    from deepCR import train
    trainer = train(image, mask, ignore=ignore, type='simulate', sky=sky, aug_sky=[-0.9, 3], name='mymodel', gpu=True,
     epoch=50, save_after=20, plot_every=10, use_tqdm=False)
    trainer.train()
    filename = trainer.save() # not necessary if save_after is specified

The aug_sky argument enables data augmentation in sky background; random sky background in the range
[aug_sky[0] * sky, aug_sky[1] * sky] is used for each input image. Sky array / sky.npy must be provided to use this functionality.
This adapts the trained model to a wider range of sky background or equivalently
exposure times. Remedy for the fact that exposure time in the training set is discrete and limited.

The save_after argument lets the trainer to save models on every epoch after save_after which has the currently lowest
validation loss. If this is not specified, you have to use trainer.save() to manually save the model at the last epoch.

After training, you can examine that validation loss has reached its minimum by

.. code-block:: python

    trainer.plot_loss()

If validation loss is still reducing, you can continue training by

.. code-block:: python

    trainer.train_phase1(20)

Do not use trainer.train(). Specify number of additional epochs.

Loading your new model
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    from deepCR import deepCR
    mdl = deepCR(mask='save_directory/my_model_epoch50.pth', hidden=32)

It's necessary to specify the number of hidden channels in the first layer if it's not default (32).

Testing your model
^^^^^^^^^^^^^^^^^^

You should test your model on a separate test set, which ideally should come from different fields than the training
set and represent a wide range of cases, e.g., exposure times. You may test your model separately on different
situations.

.. code-block:: python

    from deepCR import roc
    import matplotlib.pyplot as plt
    tpr, fpr = evaluate.roc(mdl, image=image, mask=mask, ignore=ignore)
    plt.plot(fpr, tpr)
    plt.show()

