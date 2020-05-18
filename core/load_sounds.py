import torch
import librosa
import os
from os.path import join
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import load
from librosa.feature import melspectrogram as mel
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
    def make_log_mels_from_recordings(recordings, sampling_rate, n_mels, fft_window, hop_length):
        mels = [
            np.log(mel(rec,
                sr=sampling_rate,
                n_fft=fft_window,
                hop_length=hop_length,
                n_mels=n_mels
               )) for rec in recordings
        ]
        return mels

    @classmethod
    def make_log_mels_for_classes(cls, path, classes, sampling_rate, n_mels, fft_window, hop_length):
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
                hop_length
            )
            all_mels[d] = mels
        return all_mels

    @classmethod
    def make_log_mels_from_root(cls, path, sampling_rate, n_mels, fft_window, hop_length, n_max=None):
        assert os.path.isdir(path)
        classes = [d for d in os.listdir(path) if os.path.isdir(join(path, d))]
        if n_max is not None and len(classes) > n_classes:
            classes = classes[:n_max]
        mels = cls.make_log_mels_for_classes(path, classes, sampling_rate, n_mels, fft_window, hop_length)
        return all_mels