import imp
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import os

from utils import Dataset as d
import VGGish_model as m

imp.reload(m)
imp.reload(d)


def train(model, train_loader, criterion, optimizer):
    '''
    performs forward and backward pass for a single epoch and returns average
    epoch loss and training accuracy
    '''
    model.train()
    epoch_loss = 0.
    total_correct = 0
    for i, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out = model(data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().numpy()
        pred = out.argmax(1)
        total_correct += (pred == target).sum().cpu().numpy()
    return epoch_loss / len(train_loader.sampler), \
            total_correct / len(train_loader.sampler)


def test(model, test_loader, criterion):
    '''
    performs a forward pass for a single iteration through the test data and
    reports average test loss and accuracy
    '''
    model.eval()
    test_loss = 0.
    total_correct = 0
    for i, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out = model(data)
        loss = criterion(out, target)
        test_loss += loss.detach().cpu().numpy()
        pred = out.argmax(1)
        total_correct += (pred == target).sum().cpu().numpy()
    return test_loss / len(test_loader.sampler), \
            total_correct / len(test_loader.sampler)


def run_training(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    '''
    defines training/testing routine
    '''

    train_losses = []
    test_losses = []

    print('\nstarting training for {} epochs. \nCuda is available: {}'.format(
        n_epochs, torch.cuda.is_available()))
    print('number of GPUs: {}'.format(torch.cuda.device_count()))
    print('training on GPU {}'.format(torch.cuda.current_device()))

    for e in range(n_epochs):
        print('\nEpoch:', e)
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        #print('training loss:', train_loss)
        print('training accuracy:', train_acc)
        test_loss, test_acc = test(model, test_loader, criterion)
        test_losses.append(test_loss)
        #print('test loss:', test_loss)
        print('test accuracy:', test_acc)

    return train_losses, test_losses


def main(save_weights=None):
    '''
    initialization of model, optimizer, loss function and starts training via
    run_training()
    '''

    print('initializing data, loader and model')
    data = d.BirdSoundsDataset('../data/largeBirds2_processed', max_size=200)
    train_loader, test_loader = d.make_loaders(data, batch_size=512, val_split=.2)

    weights_file = '../vggish_audioset_weights_without_fc2.h5'
    net = m.VGGish(n_bins=64, n_frames=96, n_classes=10)
    net.init_weights(weights_file)
    net.freeze_bottom()
    #new_top = nn.Linear(net.out_dims*512, net.n_classes)
    new_top = nn.Sequential(
                nn.Linear(net.out_dims*512, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, net.n_classes)
                )
    net.classifier = new_top
    if torch.cuda.is_available:
        net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

    print('starting training')
    l1, l2 = run_training(net, train_loader, test_loader, criterion, optimizer, n_epochs=100)

    if save_weights is not None:
        torch.save(net.state_dict(), os.path.join(save_weights, 'net_weights.pt'))


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        torch.cuda.current_device()
    else:
        print('No GPUs.. working on CPU')

    main()
