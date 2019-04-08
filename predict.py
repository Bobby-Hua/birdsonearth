import sys, getopt
import imp
import params as p
import VGGish_model as m
import torch
import os
import pickle
import numpy as np

from utils import vggish_input
from utils import preprocessing as pre
from utils import Dataset as d
from utils import trainer as t

imp.reload(p)
imp.reload(d)
imp.reload(m)
imp.reload(t)
imp.reload(pre)


def prepare(params):
    '''
    reads in a terminal command of the form:

    $ python predict.py <file path 1> <file path 2> ...

    and returns a list of files for inference
    TODO: prediction for directory, taking model name as option
    '''

    # reading in file names from terminal command
    print('working out options')
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'n:d:', ['name=', 'device='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit()

    for o, a in opts:
        if o in ('-n', '--name'):
            params.name = a
        if o in ('-d', '--device'):
            params.device = a

    files = None
    if args:
        files = args
    if files is None:
        raise IOError('provide a file to predict like so:\n \
                        $python predict.py <file path>')

    return params, files


def load_model_with(params):
    print('loading model')
    # load class labels
    with open(os.path.join(params.model_zoo, params.name+'.pkl'), 'rb') as f:
        labels = pickle.load(f)
    # init network and load weights
    params.n_classes = len(labels)
    device = torch.device(params.device)
    net = m.VGGish(params)
    new_top = torch.nn.Linear(net.out_dims*512, net.n_classes)
    net.classifier = new_top
    net.load_state_dict(torch.load(os.path.join(params.model_zoo, params.name+'.pt'),
                        map_location=device))
    net.to(device)
    net.eval()
    print('model for labels {} is ready'.format(labels))
    return net, labels


def predict(net, labels, files, params):
    print('starting inference')
    device = torch.device(params.device)
    predictions = []
    probs = []
    for i, file in enumerate(files):
        processed = file.split('.')[0] + '_proc.wav'
        pre.preprocess(file, processed)
        data = vggish_input.wavfile_to_examples(processed)
        data = torch.from_numpy(data).unsqueeze(1).float()
        data = data.to(device)
        net.to(device)
        out = net(data)
        mean_probs = np.mean(out.detach().cpu().numpy(), axis=0)
        pred = out.argmax(1).float()
        consensus = torch.round(torch.mean(pred))
        consensus = int(np.round(consensus.cpu().numpy()))
        print('file {} sound like a {} to me'.format(i, labels[consensus]))
        print('my guesses are: ')
        for i, label in enumerate(labels):
            print('{0}: {1:.04f}'.format(label, mean_probs[i]))
        predictions.append(labels[consensus])
        probs.append(mean_probs)
        os.remove(processed)
    return predictions, probs


if __name__ == '__main__':

    params = p.Params()
    params, files = prepare(params)
    net, labels = load_model_with(params)
    _ = predict(net, labels, files, params)
