# Welcome to the Birds on Earth repository

<!-- [![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/birds-on-mars/birdsonearth/blob/master/training_interface.ipynb) -->


In this repository we provide PyTorch code and pretrained networks to
 - classify bird geni based on their calls
 - classify the urban sounds dataset
 - use a pretrained CNN and fine-tune it on any sound classification task.

Our implementation is based on the VGG-like audio classification model [VGGish](https://github.com/DTaoo/VGGish).
We also converted the original pretrained VGGish network from Tensorflow to PyTorch. The pretrained network is based on the [AudioSet Dataset](https://research.google.com/audioset/index.html).

## Usage / Quick Start

There are a number of possible ways to use birdsonearth:
* Use the [pretrained bird classification model](https://www.google.de) to predict one of ten given bird geni: See bird_predict.ipnyb for instructions or directly use bird_predict.py.
* Use the [pretrained AudioSet model](https://drive.google.com/) as a warm start for any sound classification task: See sound_classification.ipnyb for instructions.

## Prerequisites

For training and/or prediction you will need the following Python (3.X) packages:

* `pytorch`
* `numpy`
* `pandas`
* `scipy`
* `resampy`
* `six`
* `librosa`


You can install the packages using pip & the provided `requirements.txt`

## General

All relevant parameters for training and inference are collected in `params.py`.
There are two interfaces to make use of this repo:
1. command line:
 - for training use:
 ```
 $ python -m train [options]
 ```
 - for inference use:
 ```
 $ python -m predict <file1>  <file2> ...
 ```
 For Details see below.
2. notebooks: `training_interface.ipynb` and `prediction_interface.ipynb` walk you through
 the respective procedures.

## Datasets

datasets are not included in this repo. To reproduce our trainings do the following:
- for the bird sounds:
  run
  ```
  python -m datasets.birdsounds
  ```
  to download the data that we used to train the `birds10` model with the below classes (this will take a while).
  The script `utils/bridsounds.py` provides a crawler for the [Xeno Canto](https://www.xeno-canto.org/) API.
  You can change the API-queries at the bottom to make your own dataset. Change the
  parameter `Crawler.max` to set an upper limit for the number of files to be downloaded per query.
- for the urbansounds dataset:
  [Download](https://www.kaggle.com/pavansanagapati/urban-sound-classification) the dataset.
  Unpack the `train.tar`. In `utils/urbansounds.py` change `root` to where you unpacked the data to,
  and `save_to` to where you want the preprocessed files to be stored. Then run
  ```
  $ python -m datasets.urbansounds
  ```
  This dataset consists of different wav formats which need to be resampled before creating
  spectrograms from them. This takes some time.

## Pretrained Models

In `models/` we provide weights for the urban sounds classification task (`urbansounds.pt`)
(~91% accurate) and the bird classification task (`birds10.pt`) (~92% accurate).
The respective `.pkl` files contain a list of class labels and will be processed
automatically.
Also we converted the pre-trained weights for the convolutional part of the VGGish net
for pytorch (`VGGish_convolutions.pt`).

## Details for bird geni prediction

Currently, the following bird geni are supported:
* Coloeus   -   daw     -   Dohle
* Columba   -   dove    -   Taube
* Erithacus -   robin   -   Rotkehlchen
* Garrulus  -   jay     -   Eichhäher
* Parus     -   tit     -   Meise
* Passer    -   sparrow -   Spatz
* Pica      -   magpie  -   Elster
* Picoides  -   pecker  -   Specht
* Sturnus   -   stur    -   Star
* Turdus    -   thrush  -   Drossel

The model was trained using bird calls from [Xeno Canto](https://www.xeno-canto.org/). We also provide a crawler for Xeno Canto (xeno-canto_crawler.py) in case you would like to train your own bird prediction model.

## Details for sound classification transfer learning / fine-tuning

In order to use our model as a warm start for any other sound classification task, you just need to put your sounds as mp3 or wav files in a folder with the following structure:

- repo directory
  - data
    - dataset
      - class1 (put files for class1 here)
      - class2 (put files for class2 here)
      - ...

The training script (training.py) will automatically extract your classes from folder names and split into train and val set. See training.py for possible training parameters.


## Details for Command Line Usage

- training:
  ```
  $ python -m train [options]
  ```
  may have the following options:
  - `-d` or `--data`: path to training data with the above structure
  - `-e` or `--epochs`: number of training epochs
  - `-b` or `--batchsize`: batch size for training
  - `-f` or `--format` : format of the data (only wav and mp3 are tested)
  - `-n` or `--name`: name under which model parameters will be saved

- predict:
  ```
  $ python -m predict [name] <file1> <file2> ...
  ```
  may be given a model name with the option:
  - `-n` or `--name`: will try to load the model parameters from `params.model_zoo/<name>.pt`
   and a respective labels file `".pkl`
  Files to be predicted are listed following to this.
- Note:
  - options override those specified in the `params.py` script
  - If the directory specified by `Params.mel_spec_root` exists it will be assumed that it
   contains spectrograms previously computed. If it does not spectrograms will be computed and saved
   in this directory.
  - `Params.device` specifies where computation is performed.
    By default this is set to `'cpu'` which is slow. If you have access to an nvidia GPU
    set it to `'cuda:<device index>'`, e.g. `cuda:0`.

## Details for notebook Usage

Start a jupyter server inside this repo and follow instructions
in the training and/or inference notebook.

## Model details
The VGGish model is a convolutional neural network that works with spectrograms as
input. It is a variation of the VGG network designed for image classification.
For details refer to this [publication](https://arxiv.org/abs/1609.09430).

## Contact
Please contact [Lucas](mailto:lucas@birdsonmars.com) or [Konrad](mailto:konrad@birdsonmars.com) for any questions.

## License
All code is licensed under the Apache2 license.

We use the preprocessing procedure to generate spectrograms from the original
[VGG_ish](https://github.com/tensorflow/models/tree/master/research/audioset) implementation. Our model implements the VGG_ish architecture in Pytorch.
The h5py weights file for the convolutional part is taken from the Keras implementation by [DTaoo](https://github.com/DTaoo/VGGish).
