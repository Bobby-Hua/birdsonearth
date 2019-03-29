'''
script to restructure the urban sounds data set into the required format to run
the train.py script on it.
Download the the dataset from:
https://www.kaggle.com/pavansanagapati/urban-sound-classification
Then unpack train.zip into root specified below, define where the restructured
data is saved to (save_to) and the maxium instances per class (nmax) and run
this script
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from scipy.io import wavfile
from librosa.core import load
from librosa.output import write_wav
import sys

class UrbanSoundsProcessor:

    def __init__(self, root, save_to, nmax=None):
        self.root = root
        self.nmax = nmax
        self.save_to = save_to
        self.df = pd.read_csv(os.path.join(root, 'train.csv'))
        self.labels = self.df.Class.unique()
        self.counts = {i:0 for i in self.labels}
        self.delete_download = False

    def setup(self):
        for label in self.labels:
            os.makedirs(os.path.join(self.save_to, label))

    def run(self):
        self.setup()
        total = 0
        for i in self.df.index:
            if self.nmax is not None and total >= self.nmax * len(self.labels):
                print('done loading {} instances each'.format(self.nmax))
                break
            label = self.df.at[i, 'Class']
            if self.nmax is not None and self.counts[label] >= self.nmax:
                continue
            else:
                file = str(self.df.at[i, 'ID']) + '.wav'
                print('processing ', file)
                y, sr = load(os.path.join(self.root, 'Train', file))
                #print('range: ', np.min(y), np.max(y))
                y *= 32768
                y = y.astype(np.int16)
                wavfile.write(os.path.join(self.save_to, label, file), rate=22050, data=y)
                self.counts[label] += 1
                total += 1
        if self.delete_download:
            shutil.rmtree(self.root)


if __name__ == '__main__':

    root = 'data/urbansounds'
    save_to = 'data/full_urbansounds_restructured'
    proc = UrbanSoundsProcessor(root, save_to)
    proc.run()
