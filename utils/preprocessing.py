import numpy as np
import os
import pandas as pd
from librosa.core import load
from scipy.io import wavfile
from importlib import reload
from utils import vggish_input

reload(vggish_input)


def preprocess(src, dst):
    y, sr = load(src)
    y *= 32768
    y = y.astype(np.int16)
    wavfile.write(dst, rate=22050, data=y)


def process_wavs_for_training(params):
    '''
    iterates through data under root that must have the structure
    root/{classes}/{instances}. Data must be in integer(!) .wav format.
    Under save_to the same subdirectories save_to/{classes}
    are generated and for every instance vggish compatible inputs are generated according
    to the parameters specified in vggish_params.py

    Args:
        root (string): path to data root
        saveto (string): path to root where processed data is stored
    '''
    if os.path.exists(params.mel_spec_root):
        print('spectrograms seem to have been computed')
        print('delete {} manually and rerun preprocessing or keep goint to use what is there.'.format(
        params.mel_spec_root))
        return
    labels = os.listdir(params.data_root)
    for label in os.listdir(params.data_root):
        if not os.path.isdir(os.path.join(params.data_root, label)): continue
        print('processing data for label {}'.format(label))
        os.makedirs(os.path.join(params.mel_spec_root, label))
        for file in os.listdir(os.path.join(params.data_root, label)):
            data = vggish_input.wavfile_to_examples(os.path.join(params.data_root, label, file))
            for i in range(data.shape[0]):
                np.save(os.path.join(params.mel_spec_root, label, file[:-4]+str(i)+'.npy'), data[i])


def summary(params):
    labels = [f for f in os.listdir(params.data_root)
                if os.path.isdir(os.path.join(params.data_root, f))]
    counts = [len(os.listdir(os.path.join(params.data_root, label))) for label in labels]
    df = pd.DataFrame(data={'labels':labels, 'instances':counts})
    print('the following classes and numer of instances are found:\n')
    print(df)
    processed = os.path.exists(params.mel_spec_root)
    if processed:
        print('spectrograms for the data seem to have been computed under {} \
                and will be used for training'.format(params.mel_spec_root))
    else:
        print('spectrograms for the given data need to be computed by running \
                preprocessing')
