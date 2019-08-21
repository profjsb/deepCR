---
title: 'deepCR: Cosmic Rejection with Deep Learning'
tags:
  - Python
  - Pytorch
  - astronomy
  - image processing
  - cosmic ray
  - deep learning

authors:
  - name: Keming Zhang
    orcid: 0000-0002-9870-5695
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Joshua S. Bloom
    orcid: 0000-0002-7777-216X
    affiliation: "1, 2"
affiliations:
 - name: Department of Astronomy, University of California, Berkeley
   index: 1
 - name: Lawrence Berkeley National Laboratory
   index: 2
date: 15 August 2019
bibliography: paper.bib
---

# Summary

Astronomical imaging and spectroscopy data are frequently corrupted by
"cosmic rays" (CR) which are high energy charged particles that are instrumental, 
terrestrial, or cosmic in origin. When such particles pass through solid state 
detectors, such as charged coupled devices (CCDs), they create excess 
flux in the pixels hit which lead to artifacts in images. These 
artifacts must be identified and either masked or replaced, before 
further scientific analysis could be done on the image data. It is straightforward 
to identify these artifacts when multiple exposures of the same field are 
taken. In such cases, a median image could be calculated from aligned single 
exposures, effectively creating a CR-free image. Each one of the exposures 
is then compared with the median image to identify the cosmic rays. However, 
when CCD read-out times are non-negligible, or when sources of 
interest are transient or variable, cosmic ray rejection with multiple 
exposures can be sub-optimal or infeasible. These cases would require specialized
algorithms to detect cosmic rays in single images.

![Neural network architecture of ``deepCR``. Feature 
maps are represented by gray boxes while the number of channels and example 
feature map dimensions are indicated on the top of and to the left of each 
feature map, respectively. Different computational operations are marked in 
the legend to the lower left. Unfilled boxes to the right of blue arrows represent 
feature maps directly copied from the left, which are to be concatenated with the 
adjacent feature map. To apply the inpainting model, the predicted mask (dotted box 
at left) is concatenated with the original image as the 
input.](https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/network.png)

``deepCR`` is a Python package for single frame cosmic ray rejection which is
based on deep learning and written with the Pytorch framework [@pytorch].
Since ``deepCR`` is based on deep learning, different models trained on
data taken with different instrument configurations are required, when applied to different
data. The current version of ``deepCR`` is prepackaged with model for Hubble 
Space Telescope ACS/WFC imaging data, and we expect models available 
to grow with contribution from the community. We plan to host a "model zoo"
 which enables ``deepCR`` to work across different instrument configurations.

The API of ``deepCR`` includes functionality for both applying models and 
training models. To apply an available model, ``deepCR`` takes in an input image
 and produces a cosmic ray mask and an "inpainted" image, with 
 the artifact pixels replaced with ``deepCR`` predictions. To train a new model, 
 users would feed in custom dataset to the training API, which is automated.
 ``deepCR`` works with both CPU, which is well-threaded at application time, and GPU.
 On GPU, training a new model takes as short as 20 minutes, 
 while applying ``deepCR`` on a 10 Mpix image requires less than 0.2 second, 
 orders of magnitude faster than current state of the art ``LACosmic`` [@lacosmic].

![Examples of cosmic ray contaminated image cutouts (first row),
 deepCR cosmic ray mask predictions (middle row), and images with artifact
pixels replaced with deepCR predictions 
(last row).](https://raw.githubusercontent.com/profjsb/deepCR/master/imgs/postage-sm.jpg)
 
In the paper accompanying ``deepCR`` [@deepcr], the authors showed that 
on Hubble Space Telescope (HST) ACS/WFC data, 
``deepCR`` is more robust, and at least as fast as the current 
state-of-the-art single frame cosmic ray rejection package, ``LACosmic``. The API 
of ``deepCR`` serve as a drop in replacement for ``LACosmic``, 
so that users may experiment with different packages easily. At 
reasonable false detection rates, ``deepCR`` achieved near perfect
cosmic ray detection in extragalactic and globular cluster fields, and above 
90% in more difficult dense stellar fields in nearby resolved galaxies. 
Since HST imaging is among the hardest cosmic ray rejection to be
solved, ``deepCR`` would work well across many different instrument set-ups,
including ground based imaging and spectroscopy. The combination of 
speed and accuracy of ``deepCR`` allows astronomers to potentially save
large amounts of precious observational and computational resources.

# Acknowledgements

This work was supported by a Gordon and Betty Moore Foundation Data-Driven Discovery grant, 
and has made use of the following software:

astropy [@astropy];
astrodrizzle [@astrodrizzle];
numpy [@vdw11];
scipy [@scipy];
matplotlib [@matplotlib]; 
astroscrappy [@astroscrappy];
pytorch [@pytorch];
Jupyter [@jupyter];
Scikit-image [@scikit-image]


# References