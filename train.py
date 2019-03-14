import sys, getopt
import imp
import params as p
import VGGish_model as m
import torch
import pickle
import os

from utils import Dataset as d
from utils import trainer as t
from utils import preprocessing as pre

imp.reload(p)
imp.reload(d)
imp.reload(m)
imp.reload(t)
imp.reload(pre)

def main():
    '''
    reads in a terminal command in the format:

    $ python train.py -d <data dir> -n <# training epochs> -b <batch size>

    options in this command override those in params.py.
    subsequnetly initializes a network from VGGish_model.py,
    a dataset from Dataset.py and a Trainer instance from trainer.py and runs
    '''

    params = p.Params()

    # setting up device to train on
    if torch.cuda.is_available():
        device = torch.device(params.device)
        print('GPU is accessible, working on device ', device)
    else:
        device = torch.device('cpu')
        print('No GPU accessible working on CPU')

    #reading in options from terminal command
    print('working out options')
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:n:b:', \
                            ['data=', 'nepochs=', 'batchsize='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit()

    params = p.Params()

    for o, a in opts:
        if o in ('-d', '--data'):
            params.data_root = a
        if o in ('-n', '--nepochs'):
            params.n_epochs = int(a)
        if o in ('-b', '--batchsize'):
            params.batch_size = int(a)

    # Need to be fixed - data format is set to mp3 as default
    '''
    if params.data_format == 'mp3':
        #TODO: not yet tested
        pre.process_mp3_for_training(params.data_root)
        params.data_root += '_processed'
    '''

    # initialization for training
    print('setup')
    dataset = d.BirdSoundsDataset(params)
    net = m.VGGish(params)
    net.init_weights()
    net.freeze_bottom()
    new_top = torch.nn.Linear(net.out_dims*512, net.n_classes)
    net.classifier = new_top
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    trainer = t.Trainer(net, dataset, criterion, optimizer, params, device)

    # starting training
    print('start training on ', device)
    trainer.run_training()

    # saving model weights and class labels
    print('saving weights and class labels')
    net.save_weights()
    print(dataset.labels)
    with open(os.path.join(params.model_zoo, params.name+'.pkl'), 'wb') as f:
        pickle.dump(dataset.labels, f)


if __name__ == '__main__':
    main()
