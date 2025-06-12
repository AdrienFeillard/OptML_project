# CS-439 : Mini-Project 

# The biases behind rating: Uncovering the hidden influences in beer ratings

## Quickstart

```bash
# clone project
git clone git@github.com:AdrienFeillard/OptML_project.git
cd ada-2024-project-insightmakers24

# [OPTIONAL] create conda environment
conda create -n AdaptativeNoise
conda activate AdaptativeNoise

# install requirements
pip install -r requirements.txt
```

### How to use the library

- The `train.py` script is used for individual training runs. It is highly configurable via command-line arguments listed at the end of this document.

- `run_experiment.py` file is parametrisable, and is used to launch many training sequencially. It will always run the baseline for each experiments.

- A directory `checkpoints` is generated, it contains the checkpoint for each experiences launches and a `metrics` directory that contains many JSON of metrics of each experiment (all contained in `all_metrics.json`)


## Project Structure

The directory structure of the project looks like this:

```
.
│
├── checkpoints/                              <- Generated directory : contains the checkpoints for each experiments
│   └── metrics/                              <- Generated directory : contains JSONs of metrics for each experiments
│
├── core/
│   ├── cifar10_models/
│   │   ├── SimpleCNN.py                      <- Custom CNN architecture of different depth
│   │   ├── resnet.py                         <- ResNet architecture (e.g., Resnet18)
│   │   └── ...                     
│   ├── data.py                               <- Handles CIFAR-10 dataset loading and preprocessing
│   ├── module.py                             <- Core module for neural network components
│   ├── noise_regularization.py               <- Implements the adaptive noise injection strategy and regularization
│   └── schduler.py                           <- Learning rate scheduler or similar optimization scheduling logic
│
├── notebooks/                                <- Directory for Jupyter notebooks (e.g., `plot_adrien.ipynb`)
│
├── run_experiment.py                         <- General script for launching parametrisable experiments 
├── test.py                                   <- Script for evaluating trained models
├── train.py                                  <- Core training loop script
│
└── utils/                                    <- Utility functions and configuration management
```

## Abstract

This paper explores the impact of noise injection on
Convolutional Neural Networks trained for image classification
on the CIFAR-10 dataset.
Despite advances in machine learning, some challenges like
poor convergence to optimal solutions and model overfitting
persist, limiting the practical application and reliability of CNNs
The injection of noise can be used to address both these issues.
This study investigates the role of the noise in helping the
model to converge to optimal solution by escaping local minima
or saddle points and also address the impact of the noise as a
regularizer and on the generalization of the models.
Our solution is to inject an adaptive noise, that is activated
and scaled using the model parameters, such as the loss or the
gradient. The goal is to inject the noise only at specific times,
when the model might be stuck on a local minimum or might be
overfitting. We tried this Noise injection on four different models,
using different optimizers and noise types.


### Command-Line Arguments for `train.py`

The `train.py` script offers extensive parametrization to control experiments. Below are the key arguments you can use:

#### General Training Parameters:

* `--data-dir` (str, default: `./data/cifar10`): Path to the CIFAR-10 dataset directory.
* `--classifier`, `-c` (str, default: `resnet18`): Classifier model to use. Options include `resnet18`, `SimpleCNN`, `TinyCNN`, `BabyCNN`.
* `--batch-size`, `-b` (int, default: `128`): Batch size for training.
* `--epochs`, `-e` (int, default: `100`): Maximum number of training epochs.
* `--workers`, `-n` (int, default: `4`): Number of data loading workers.
* `--subset`, `-s` (float, default: `1.0`): Fraction of dataset to use for training (0.0-1.0).
* `--lr` (float, default: `1e-2`): Initial learning rate.
* `--wd` (float, default: `1e-3`): Weight decay (L2 regularization strength).
* `--gpu`, `-g` (str, default: `"0"`): GPU ID(s) to use (e.g., "0", "0,1,2").
* `--save-as` (str, default: `none.pth`): Custom filename to save the model checkpoint.
* `--download-weights`, `-w` (bool, default: `False`): Download pre-trained weights for classifiers (if available).

#### Optimizer Parameters:

* `--optimizer`, `-o` (str, default: `SGD`): Optimizer to use. Options: `SGD`, `Adam`.
* `--momentum`, `-m` (float, default: `0.9`): Momentum factor for SGD optimizer.
* `--beta1`, `-b1` (float, default: `0.9`): Beta1 parameter for Adam optimizer.
* `--beta2`, `-b2` (float, default: `0.999`): Beta2 parameter for Adam optimizer.
* `--lr-restart-period` (int, default: `50`): Period for learning rate restarts (if applicable to scheduler).

#### Noise Injection Parameters:

* `--noise-magnitude`, `-nm` (float, default: `0.01`): Initial magnitude (scale) of the injected noise.
* `--noise-schedule`, `-ns` (str, default: `constant`): Schedule for noise magnitude over time.
* `--noise-distribution`, `-nd` (str, default: `gaussian`): Distribution of noise (`gaussian` or `uniform`).
* `--noise-layer`, `-nl` (list of str, default: `None`): List of specific layer names to apply noise to (e.g., `--noise-layer conv1 --noise-layer fc`). If `None`, noise is applied globally.
* `--permanent`, `-p` (bool, default: `False`): If `True`, applies noise permanently to weights (not just during forward pass).

#### Adaptive Noise Control Parameters:

* `--noise-during-stuck-only`, `-so` (bool, default: `False`): **Key parameter for adaptive noise.** If `True`, noise is applied only when training is detected as 'stuck' or overfitting.
* `--patience` (int, default: `5`): Number of epochs to wait before applying noise when training is detected as stuck.
* `--flag-window-size` (int, default: `5`): Number of epochs for the history window used to detect flags (e.g., for loss plateau).
* `--flag-min-epochs` (int, default: `10`): Minimum number of epochs before the adaptive flag system starts checking.
* `--flag-overfit-epochs` (int, default: `3`): Number of consecutive epochs validation loss must increase to trigger the overfitting flag.
* `--flag-plateau-delta` (float, default: `1e-4`): Minimum improvement in loss/metric to not be considered a plateau.
* `--flag-grad-plateau-thr` (float, default: `0.05`): Threshold for relative gradient norm improvement to detect a plateau (e.g., 0.05 for 5%).
* `--flag-low-weight-update-thr` (float, default: `1e-4`): Threshold for detecting low weight update norm, indicating potential stagnation.
* `--disable-adaptive-flags` (bool, default: `False`): If `True`, completely disables the adaptive flag system, making noise injection static.
* `--relative-min-noise` (float, default: `0.01`): Minimum relative magnitude for adaptive noise scaling.
* `--relative-max-noise` (float, default: `0.10`): Maximum relative magnitude for adaptive noise scaling.
* `--consecutive-flag-trigger` (int, default: `3`): Number of consecutive flag triggers required before noise is activated.
* `--min-cooldown-epochs` (int, default: `5`): Minimum number of epochs to wait after noise injection before re-evaluating flags.
* `--max-cooldown-epochs` (int, default: `15`): Maximum number of epochs for cooldown after noise injection.
* `--visualize-lr` (bool, default: `False`): Visualize the learning rate schedule.
* `--disable-graphs` (bool, default: `True`): Disable ASCII graphs in the dashboard (for console output).
* `--detailed-metrics` (bool, default: `False`): Enable slower, detailed per-class metric collection for the training set.