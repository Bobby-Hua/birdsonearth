# Welcome to the Birds on Earth repository

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

All relevant parameters for training and inference are collected in params.py.
There are two interfaces to make use of this repo:
1. command line:
 - for training use:
 ```
 $ python -m train [options]
 ```
 Note: options in this command override those in params.py
 - for inference use:
 ```
 $ python -m predict <file1>  <file2> ...
 ```
2. notebooks: training_interface and prediction_interface walk you through
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
 and `save_to` to where you want the preprocessed file to be stored. Then run
 ```
 $ python -m datasets.urbansounds
 ```
 This dataset consists of different wav formats which need to be resampled before creating
 spectrograms from them. This takes some time.

## Pretrained Models

In `models/` we provide weights for the urban sounds classification task (`urbansounds.pt`)
and the bird classification task (`birds10.pt`).
The respective `.pkl` files contain a list of class labels and will be processed
automatically.
Also we converted the pre-trained weights for the convolutional part of the VGGish net
for pytorch (`VGGish_convolutions.pt`).

## Details for bird geni prediction

Currently, the following bird geni are supported:
* Coloeus   -   daw     -   Dohle
* Columba   -   dove    -   Taume
* Erithacus -   robin   -   Rotkehlchen
* Garrulus  -   jay     -   Eichhäher
* Parus     -   tit     -   Meise
* Passer    -   sparrow -   Spatz
* Pica      -   magpie  -   Elster
* Picoides  -   pecker  -   Specht
* Sturnus   -   stur    -   Star
* Turdus    -   thrush  -   Drossel

The model was trained using bird calls from [Xeno Canto](https://www.xeno-canto.org/). We also provided a crawler for Xeno Canto (xeno-canto_crawler.py) in case you would like to train your own bird prediction model.

## Details for sound classification transfer learning / fine-tuning

In order to use our model as a warm start for any other sound classification task, you just need to put your sounds as mp3 files in a folder with the following structure:

- repo directory
  - data
    - dataset
      - class1 (put files for class1 here)
      - class2 (put files for class2 here)
      - ...

The training script (training.py) will automatically extract your classes from folder names and split into train and val set. See training.py for possible training parameters.

## Model details
tbd

## Contact
Please contact [Lucas](mailto:lucas@birdsonmars.com) or [Konrad](mailto:lucas@birdsonmars.com) for any questions.

## License
All code is licensed under the Apache2 license.
