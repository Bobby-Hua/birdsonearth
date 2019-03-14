# Welcome to the Birds on Earth repository

In this repository we provide PyTorch code and pretrained networks to
 - classify bird geni based on their calls and/or
 - use a pretrained CNN and fine-tune it on any sound classification task.
 
Our implementation is based on the VGG-like audio classification model [VGGish](https://github.com/DTaoo/VGGish).
We also converted the original pretrained VGGish network from Tensorflow to PyTorch. The pretrained network is based on the [AudioSet Dataset](https://research.google.com/audioset/index.html).

## Usage / Quick Start

There are two possible ways to use birdsonearth:
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

## Details for bird geni prediction

Currently, the following bird geni are supported: 
* Spatz = Passer
* Taube = Columba
* Drossel = Turdus
* Meise = Cyanistes
* Sittich = Psittacula
* what else?

The model was trained using bird calls from [Xeno Canto](https://www.xeno-canto.org/). We also provided a crawler for Xeno Canto (xeno-canto_crawler.py) in case you would like to train your own bird prediction model.

## Details for sound classification transfer learning / fine-tuning

In order to use our model as a warm start for any other sound classification task, you just need to put your sounds as mp3 files in a folder with the following structure:

|repo directory <br>
|⋅⋅⋅data <br>
|⋅⋅⋅⋅⋅⋅classname1 (put your mp3 files for class1 here)<br>
|⋅⋅⋅⋅⋅⋅classname2 (put your mp3 files for class2 here)<br>
|⋅⋅⋅⋅⋅⋅and so on<br>
   
The training script (training.py) will automatically extract your classes from folder names and split into train and val set. See training.py for possible training parameters. 

## Model details
tbd

## Contact
Please contact [Lucas](mailto:lucas@birdsonmars.com) or [Konrad](mailto:lucas@birdsonmars.com) for any questions.

## License
All code is licensed under the Apache2 license.
