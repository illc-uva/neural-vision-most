# Neural Models of the Psychosemantics of 'Most'

This repository accompanites the paper "Neural Models of the Psychosemantics of 'most'": .  It trains neural models of visual 
attention on the experimental paradigm of Pietroski et al 2009, and contains results roughly matching human performance and
a discussion of what to infer from this.

## Requirements

The experiments were run using
- Python 3.5.0
- CUDA 9.0.176 / cuDNN 7.0.5
- TensorFlow 1.8.0

Later versions of the above are likely to work, but your mileage may vary.  The code uses TF's Estimators API, which has changed
somewhat, and could be a source of errors.

## Running Trials/Experiments

## Analyzing Results

## Image Generator

While we provide train/val/test sets of images, they were all generated via `image_generator.py`.  The docstring at the top of that file contains more information about how to use it, should you want to generate new datasets manipulated in various ways (different size images, different ratios/cardinalities, more colors, etc.).
