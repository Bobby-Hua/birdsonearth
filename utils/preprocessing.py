import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import imp
from librosa.core import load
from librosa.output import write_wav
from scipy.io import wavfile

from utils import vggish_input


imp.reload(vggish_input)


def process_xeno_canto_downloads(root, save_to):
    '''
    iterates through data under root that must have the structure
    root/{classes}/{instances}. Under save_to the same subdirectories save_to/{classes}
    are generated and for every instance vggish compatible inputs are generated according
    to the parameters specified in vggish_params.py

    Args:
        root (string): path to data root
        saveto (string): path to root where processed data is stored
    '''
    labels = os.listdir(root)
    for label in os.listdir(root):
        if not os.path.isdir(os.path.join(root, label)): continue
        print('prcossing data for label {}'.format(label))
        os.makedirs(os.path.join(save_to, label))
        for file in os.listdir(os.path.join(root, label)):
            print('processing:', os.path.join(root, label, file))
            data = vggish_input.wavfile_to_examples(os.path.join(root, label, file))
            for i in range(data.shape[0]):
                np.save(os.path.join(save_to, label, file[:-4]+str(i)+'.npy'), data[i])


def process_mp3s_for_training(root):
    # TODO: not yet tested
    save_to = root + '_processed'
    labels = os.listdir(root)
    for label in os.listdir(root):
        if not os.path.isdir(os.path.join(root, label)): continue
        print('prcossing data for label {}'.format(label))
        os.makedirs(os.path.join(save_to, label))
        for file in os.listdir(os.path.join(root, label)):
            #print('processing:', os.path.join(root, label, file))
            y, sr = load(file)
            y *= 32768
            y = y.astype(np.int16)
            # pretty inefficient ...
            wavfile.write(os.path.join(root, label, 'processing.wav'), rate=22050, data=y)
            data = vggish_input.wavfile_to_examples(os.path.join(root, label, file))
            for i in range(data.shape[0]):
                np.save(os.path.join(save_to, label, file[:-4]+str(i)+'.npy'), data[i])
            os.remove(os.path.join(root, label, 'processing.wav'))


def mp3_to_wav(file):
    y, sr = load(file)
    y *= 32768
    y = y.astype(np.int16)
    filename = file.split('.')[0] + '.wav'
    wavfile.write(filename, rate=22050, data=y)
    return filename
