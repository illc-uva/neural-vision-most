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

The basic script for training and evaluating a model (as well as getting predictions) on the task is `run.py`.  It has many command-line arguments, which you can see via `python run.py --help` or by reading the bottom of the file.

An example for training a RAM (recurrent model of visual attention) with 24 glimpses and early-stopping, and evaluating the best-performing model:

```
python run.py --model ram --num_glimpses 24 --out_path . --img_size 128 --grayscale \
      --train_images images/small/train/*.png --test_images images/small/test/*.png --val_images images/small/val/*.png \
      --num_epochs 200 --epochs_per_eval 2
```

This will save files (including checkpoints) to `./ram`.  `train_eval.csv` contains validation loss and accuracy over training, while `test_eval.csv` evalautes the best model. The best model is saved to `./ram/best`.

To predict, using the best model:

```
python run.py --no_train --no_eval --predict --model ram --num_glimpses 24 --out_path ./ram/best --img_size 128 --grayscale \
      --train_images images/small/train/*.png --test_images images/small/test/*.png --val_images images/small/val/*.png
```

This will generate a file `./ram/best/test_predict.csv` containing the predictions, and other information about the images.

**N.B.:** note the first three command-line flags, which control the mode for the model.  Currently, prediction can only happen by re-running `run.py` _after_ training.  This is a small bug that I will fix at some point.

## Analyzing Results

The results of our experiment are in the `results` directory.  To reproduce what we report in the paper:

1. Generate mean accuracy data from raw data: `python process_results.py`
2. Descriptive plots:
    * `python plots.py`
    * `python learning_curves.py`
3. Regressions: `python regression.py`
4. ANS model fits: `python weber_fit.py`

## Image Generator

While we provide train/val/test sets of images, they were all generated via `image_generator.py`.  The docstring at the top of that file contains more information about how to use it, should you want to generate new datasets manipulated in various ways (different size images, different ratios/cardinalities, more colors, etc.).
