import torch
import librosa
import os
from os.path import join
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import load
from librosa.feature import melspectrogram as melspec
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')


class SoundLoader:

    def __init__(self):
        return

    @staticmethod
    def load_sounds_from_dir(path):
        assert os.path.isdir(path)
        recordings = []
        for file in os.listdir(path):
            a, _ = librosa.core.load(join(path, file))
            recordings.append(a)
        return recordings

    @staticmethod
    def make_log_mels_from_recordings(recordings, sampling_rate, n_mels, fft_window, hop_length, epsilon):
        mels = []
        for rec in recordings:
            mel = melspec(
                rec,
                sr=sampling_rate,
                n_fft=fft_window,
                hop_length=hop_length,
                n_mels=n_mels
            )
            mels.append(np.log(mel + np.array([epsilon])))
        return mels

    @classmethod
    def make_log_mels_for_classes(cls, path, classes, sampling_rate, n_mels, fft_window, hop_length, epsilon):
        assert os.path.isdir(path)
        all_mels = {}
        for i, d in tqdm(enumerate(classes)):
            if not os.path.isdir(join(path, d)):
                continue
            recordings = cls.load_sounds_from_dir(join(path, d))
            mels = cls.make_log_mels_from_recordings(
                recordings,
                sampling_rate,
                n_mels,
                fft_window,
                hop_length,
                epsilon
            )
            all_mels[d] = mels
        return all_mels

    @classmethod
    def make_log_mels_from_root(cls, path, sampling_rate, n_mels, fft_window, hop_length, epsilon, n_max=None):
        assert os.path.isdir(path)
        classes = [d for d in os.listdir(path) if os.path.isdir(join(path, d))]
        if n_max is not None and len(classes) > n_classes:
            classes = classes[:n_max]
        mels = cls.make_log_mels_for_classes(path, classes, sampling_rate, n_mels, fft_window, hop_length, epsilon)
        return all_mels

    @staticmethod
    def save_mels(mels, dst):
        if not os.path.exists(dst):
            os.mkdir(dst)
        for k, v in mels.items():
            with open(join(dst, f'{k}.pkl'), 'wb') as f:
                pickle.dump(v, f)

    @staticmethod
    def load_mels(src, classes):
        all_mels = {}
        for c in classes:
            mel = pickle.load(
                open(join(src, f'{c}.pkl'), 'rb')
            )
            all_mels[c] = mel
        return all_mels


if __name__ == '__main__':

    from utils.params import load_params
    params = load_params('core/params.yml')
    data_root = '/hdd/sounds/ESC-50'
    classes = ['101 - Dog', '102 - Rooster']#, '103 - Pig', '104 - Cow', '105 - Frog']

    all_mels = SoundLoader.make_log_mels_for_classes(
        path=data_root,
        classes=classes,
        sampling_rate=params.specs.sampling_rate,
        n_mels=params.specs.n_mels,
        fft_window=params.specs.ft_windows,
        hop_length=params.specs.hop_length,
        epsilon=params.specs.epsilon
        n_max=5)

    SoundLoader.save_mels(all_mels, '/hdd/sounds/melspecs')



