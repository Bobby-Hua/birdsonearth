import sys, getopt
import imp
import params as p
import VGGish_model as m
import torch
import os
import pickle

from utils import vggish_input
from utils import preprocessing as pre
from utils import Dataset as d
from utils import trainer as t

imp.reload(p)
imp.reload(d)
imp.reload(m)
imp.reload(t)
imp.reload(pre)


def main():
    '''
    reads in a terminal command of the form:

    $ python predict.py <file path 1> <file path 2> ...

    loads a trained network and class labels from the directory params.weights_to
    and makes predictions for the given files
    '''
    params = p.Params()

    # setting up device to calculate prediction on
    if torch.cuda.is_available():
        device = torch.device(params.device)
        print('GPU is accessible, working on device ', device)
    else:
        device = torch.device('cpu')
        print('No GPU accessible working on CPU')

    # reading in file names from terminal command
    print('working out options')
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', [])
    except getopt.GetoptError as err:
        print(err)
        sys.exit()

    files = None
    if args:
        files = args
    #TODO: add prediction for a directory
    if files is None:
        raise IOError('provide a file to predict like so:\n \
                        $python predict.py <file path>')

    # init network and load weights
    net = m.VGGish(params)
    new_top = torch.nn.Linear(net.out_dims*512, net.n_classes)
    net.classifier = new_top
    net.load_state_dict(torch.load(os.path.join(params.model_zoo, params.name+'.pt'),
                        map_location=device))
    net.to(device)
    net.eval()

    # load class labels
    with open(os.path.join(params.model_zoo, params.name+'.pkl'), 'rb') as f:
        labels = pickle.load(f)

    # predict
    for i, file in enumerate(files):
        delete = False
        if file.endswith('.mp3'):
            delete = True
            print('converting mp3 to wav')
            file = pre.mp3_to_wav(file)
        data = vggish_input.wavfile_to_examples(file)
        data = torch.from_numpy(data).unsqueeze(1).float()
        data.to(device)
        out = net(data)
        pred = out.argmax(1).float()
        consensus = torch.round(torch.mean(pred))
        consensus = int(consensus.cpu())
        print('file {} sound like a {} to me'.format(i, labels[consensus]))
        if delete:
            os.remove(file)

if __name__ == '__main__':

    main()
