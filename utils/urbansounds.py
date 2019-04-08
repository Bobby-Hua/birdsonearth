'''
script to restructure the urban sounds data set into the required format to run
the train.py script on it.
Download the the dataset from:
https://www.kaggle.com/pavansanagapati/urban-sound-classification
Then unpack train.zip into root specified below, define where the restructured
data is saved to (save_to) and the maxium instances per class (nmax) and run
this script

TODO: apparently there are different bit depths in the dataset and 24bit leads to issues
    now these files are simply ignored...
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from scipy.io import wavfile

class UrbanSoundsProcessor:

    def __init__(self, root, nmax, save_to):
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
            if total >= self.nmax * len(self.labels):
                print('done loading {} instances each'.format(self.nmax))
                break
            label = self.df.at[i, 'Class']
            if self.nmax is not None and self.counts[label] >= self.nmax:
                continue
            else:
                file = str(self.df.at[i, 'ID']) + '.wav'
                print('\n', file)
                try:
                    sr, wav_data = wavfile.read(os.path.join(self.root, 'Train', file))
                except Exception as e:
                    print(e)
                    print('not using file ', file)
                    continue
                print(type(wav_data), 'original sample rate: ', sr)
                print(np.min(wav_data), np.max(wav_data))
                wav_data = wav_data.astype(np.int16)
                wavfile.write(os.path.join(self.save_to, label, file), rate=22050, data=wav_data)
                # shutil.copyfile(os.path.join(self.root, 'Train', file),
                #             os.path.join(self.save_to, label, file))
                self.counts[label] += 1
                total += 1
        if self.delete_download:
            shutil.rmtree(self.root)


if __name__ == '__main__':

    print(os.listdir('data/'))
    root = 'data/urbansounds_download'
    save_to = 'data/urbansounds'
    nmax = 100
    proc = UrbanSoundsProcessor(root, nmax, save_to)
    proc.run()
