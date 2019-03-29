import shutil
import os
from utils.preprocessing import preprocess

root = '../data/largeBirds'
new_root = '../data/largeBirdsNew'

for dir in os.listdir(root):
    os.makedirs(os.path.join(new_root, dir))
    print('processing ', dir)
    for inst in os.listdir(os.path.join(root, dir)):
        file = os.path.join(root, dir, inst, 'sound.mp3')
        if os.path.exists(file):
            preprocess(file, os.path.join(new_root, dir, inst+'.wav'))
        else:
            print('no file for ', os.path.join(file))
