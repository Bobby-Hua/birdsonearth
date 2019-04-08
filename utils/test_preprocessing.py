from scipy.io import wavfile
from librosa.core import load
from librosa.output import write_wav
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from utils import vggish_input
import imp
imp.reload(vggish_input)

path = 'example/sound.mp3'
y, sr = load(os.path.join(path))
y *= 32768 # wav in float32 has amplitude 1 but wav in int16 has amplitude 32768
print(np.min(y), np.max(y))
y = y.astype(np.int16)
print(np.min(y), np.max(y))
#print(y.shape)
wavfile.write(path.replace('.mp3', '.wav'), rate=22050, data=y)
sr, wav_data = wavfile.read(path.replace('.mp3', '.wav'))
print(sr)
data = vggish_input.wavfile_to_examples(path.replace('.mp3', '.wav'))
print(data.shape)

for i in range(data.shape[0]):
    np.save(os.path.join('example', 'chunk{}.npy'.format(i)), data[i])
    scipy.misc.imsave(os.path.join('example', 'chunk{}.jpg'.format(i)), data[i])
